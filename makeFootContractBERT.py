import os
import os.path as osp
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import time
import random
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

### (새로 추가) matplotlib 임포트
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.utils.utils_mesh import flip_thetas_batch
from lib.data.dataset_wild import WildDetDataset
# from lib.model.loss import *
from lib.model.model_mesh import MeshRegressor
# from lib.utils.vismo import render_and_save  # 필요 없으므로 제거
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares
import pdb


#######################################################
# (1) 3D 각도 계산 (T-포즈 검출용)
#######################################################
def angle_3pts_3d(a, b, c):
    """
    3D 점 (a, b, c)에서 b를 중심으로 a→b, c→b 벡터가 이루는 각도(도수) 계산
    """
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_ = dot / (mag_ba * mag_bc)
    cos_ = np.clip(cos_, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_))
    return angle

#######################################################
# (2) scale matching용 (원본 코드 그대로)
#######################################################
def err(p, x, y):
    return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()

def solve_scale(x, y):
    print('Estimating camera transformation.')
    best_res = 100000
    best_scale = None
    for init_scale in tqdm(range(0,2000,5)):
        p0 = [init_scale, 0.0, 0.0, 0.0]
        est = least_squares(err, p0, args=(x.reshape(-1,3), y.reshape(-1,3)))
        if est['fun'] < best_res:
            best_res = est['fun']
            best_scale = est['x'][0]
    print('Pose matching error = %.2f mm.' % best_res)
    return best_scale

#######################################################
# (3) SMPL 파라미터 저장 (원본 코드 그대로)
#######################################################
def export_smpl_parameters(smpl_output, output_path, J_regressor, smpl):
    """Export SMPL parameters in a format compatible with the lab's code."""
    vertices = smpl_output.vertices.detach().cpu().numpy()
    joints = smpl_output.joints.detach().cpu().numpy()
    
    body_pose = smpl_output.body_pose.detach().cpu().numpy()
    global_orient = smpl_output.global_orient.detach().cpu().numpy()
    betas = smpl_output.betas.detach().cpu().numpy()
    faces = smpl.faces
    
    kintree_table = np.array([
        [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 12, 13, 14], 
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]   
    ], dtype=np.int32)
    J = np.matmul(J_regressor.cpu().numpy(), vertices[0])
    
    smpl_data = {
        'kintree_table': kintree_table,
        'J': joints,
        'betas': betas[0],
        'v_template': vertices[0],
        'weights': np.zeros((vertices.shape[1], 17)),
        'shapedirs': np.zeros((vertices.shape[1], 3, 10)),
        'J_regressor': J_regressor.cpu().numpy(),
        'poses': np.concatenate([global_orient, body_pose], axis=-1),
        'f': faces
    }
    
    np.savez(output_path, **smpl_data)
    print(f"SMPL parameters exported to {output_path}")


#######################################################
# (4) argparse 부분
#######################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mesh/MB_ft_pw3d.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--ref_3d_motion_path', type=str, default=None, help='3D motion path')
    parser.add_argument('--pixel', action='store_true', help='align with pixel coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

#######################################################
# (5) matplotlib 3D 시각화 함수 (새로 추가)
#######################################################
def plot_foot_3d(foot_points_3d, floor_y=None):
    """
    foot_points_3d: shape (N,3) 
    floor_y: 바닥 높이를 y=constant 로 그릴 때 사용 (None이면 그리지 않음)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 발 위치를 파란 점으로 표시
    ax.scatter(foot_points_3d[:,0], foot_points_3d[:,1], foot_points_3d[:,2], 
               c='b', marker='o', label='foot')

    if floor_y is not None:
        # 발 위치 범위를 기준으로 x,z 그리드 생성
        x_min, x_max = foot_points_3d[:,0].min()-0.5, foot_points_3d[:,0].max()+0.5
        z_min, z_max = foot_points_3d[:,2].min()-0.5, foot_points_3d[:,2].max()+0.5
        X, Z = np.meshgrid(np.linspace(x_min, x_max, 20),
                           np.linspace(z_min, z_max, 20))
        Y = np.full_like(X, floor_y)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='r', label='floor')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Foot Points & Floor (3D)")
    plt.legend()
    plt.show()


#######################################################
# (main) 전체 파이프라인
#######################################################
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    # (A) SMPL & Model 로드
    smpl = SMPL(args.data_root, batch_size=1).cuda()
    J_regressor = smpl.J_regressor_h36m

    end = time.time()
    model_backbone = load_backbone(args)
    print(f'init backbone time: {(time.time()-end):.2f}s')
    end = time.time()

    model = MeshRegressor(
        args,
        backbone=model_backbone,
        dim_rep=args.dim_rep,
        hidden_dim=args.hidden_dim,
        dropout_ratio=args.dropout
    )
    print(f'init whole model time: {(time.time()-end):.2f}s')

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    chk_filename = opts.evaluate if opts.evaluate else None
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    # (B) WildDetDataset 로더 (AlphaPose 2D → (B,T,K*2))
    testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
    }

    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    # 결과 저장 (굳이 필요 없으면 안 써도 됨)
    verts_all = []
    reg3d_all = []

    ##############################################
    # (C) T-포즈, 바닥 높이 추정 관련 변수
    ##############################################
    T_POSE_ANGLE_THRESHOLD = 30
    T_POSE_DETECT_COUNT = 5

    # SMPL->H36M 조인트 인덱스 (예시)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER= 12
    LEFT_WRIST    = 15
    RIGHT_WRIST   = 16
    LEFT_ANKLE    = 7
    RIGHT_ANKLE   = 6

    state = "INIT"
    t_pose_detected_count = 0
    t_pose_body_height = None

    foot_points_3d = []  # (x,y,z) 모으기
    frame_idx = 0

    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            batch_size, clip_frames = batch_input.shape[:2]

            # 1) forward
            if torch.cuda.is_available():
                batch_input = batch_input.float().cuda()

            output = model(batch_input)
            # flip augmentation
            batch_input_flip = flip_data(batch_input)
            output_flip = model(batch_input_flip)

            output_flip_pose  = output_flip[0]['theta'][:, :, :72]
            output_flip_shape = output_flip[0]['theta'][:, :, 72:]
            output_flip_pose  = flip_thetas_batch(output_flip_pose)
            output_flip_pose  = output_flip_pose.reshape(-1, 72)
            output_flip_shape = output_flip_shape.reshape(-1, 10)

            # 2) SMPL로부터 3D 버텍스/조인트
            output_flip_smpl = smpl(
                betas=output_flip_shape,
                body_pose=output_flip_pose[:, 3:],
                global_orient=output_flip_pose[:, :3],
                pose2rot=True
            )
            output_flip_verts = output_flip_smpl.vertices.detach()

            # 3) 17개 관절(H36M) 계산
            J_regressor_batch = J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
            output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)

            # 4) flip/no-flip 결과 평균
            output_flip_back = [{
                'verts': output_flip_verts.reshape(batch_size, clip_frames, -1, 3)*1000.0,
                'kp_3d': output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3)
            }]
            output_final = [{}]
            for k,v in output_flip_back[0].items():
                output_final[0][k] = (output[0][k] + v)/2.0

            output = output_final
            verts_all.append(output[0]['verts'].cpu().numpy())
            reg3d_all.append(output[0]['kp_3d'].cpu().numpy())

            # (선택) SMPL 파라미터 저장
            out_npz = osp.join(opts.out_path, f"my_smpl_params_{frame_idx}.npz")
            export_smpl_parameters(output_flip_smpl, out_npz, J_regressor, smpl)

            # 5) T-포즈, 바닥 높이
            kp3d_np = output[0]['kp_3d'].cpu().numpy()  # (1,clip_frames,17,3)

            for f in range(clip_frames):
                cur_joints = kp3d_np[0,f]  # (17,3)
                left_shoulder  = cur_joints[LEFT_SHOULDER]
                right_shoulder = cur_joints[RIGHT_SHOULDER]
                left_wrist     = cur_joints[LEFT_WRIST]
                right_wrist    = cur_joints[RIGHT_WRIST]

                left_arm_angle  = angle_3pts_3d(left_wrist, left_shoulder, right_shoulder)
                right_arm_angle = angle_3pts_3d(right_wrist,right_shoulder,left_shoulder)
                shoulder_diff_y = abs(left_shoulder[1] - right_shoulder[1])

                T_pose_condition = (
                    shoulder_diff_y < 50
                    and abs(left_arm_angle-180)<T_POSE_ANGLE_THRESHOLD
                    and abs(right_arm_angle-180)<T_POSE_ANGLE_THRESHOLD
                )

                if T_pose_condition:
                    t_pose_detected_count += 1
                else:
                    t_pose_detected_count = 0

                if state == "INIT" and t_pose_detected_count >= T_POSE_DETECT_COUNT:
                    state = "T_POSE_DETECT"
                    # 어깨->발목 높이
                    left_ankle  = cur_joints[LEFT_ANKLE]
                    right_ankle = cur_joints[RIGHT_ANKLE]
                    avg_ankle_y = (left_ankle[1] + right_ankle[1])/2.0
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1])/2.0
                    t_pose_body_height = abs(avg_shoulder_y - avg_ankle_y)

                    print(f"=== T-POSE DETECTED at frame={frame_idx}, body_height={t_pose_body_height:.2f} ===")
                    state = "WALKING_DETECT"

                if state == "WALKING_DETECT":
                    left_ankle  = cur_joints[LEFT_ANKLE]
                    right_ankle = cur_joints[RIGHT_ANKLE]
                    foot_mid = (left_ankle+right_ankle)/2.0  # (x,y,z)
                    foot_points_3d.append(foot_mid)

                frame_idx += 1


    # (모든 batch 처리 끝)
    foot_points_3d = np.array(foot_points_3d)
    if len(foot_points_3d) == 0:
        print("[WARN] No foot points found. Floor not estimated.")
    else:
        avg_floor_y = foot_points_3d[:,1].mean()
        print(f"[INFO] Estimated floor y={avg_floor_y:.2f}")
        if t_pose_body_height is not None and t_pose_body_height>0:
            # ex) 1 unit ~ 1700 mm
            mm_per_unit = 1700.0 / t_pose_body_height
            print(f"[INFO] => mm_per_unit={mm_per_unit:.2f}, floor_height_mm={avg_floor_y*mm_per_unit:.2f}")
        else:
            print("[INFO] T-pose not found or body_height invalid => can't scale floor in mm")

        # === (중요) matplotlib로 3D 표시 ===
        # foot_points_3d: shape (N,3)
        plot_foot_3d(foot_points_3d, floor_y=avg_floor_y)

    # (옵션) 만약 ref_3d_motion_path 있으면 scale matching
    if opts.ref_3d_motion_path:
        ref_pose = np.load(opts.ref_3d_motion_path)
        # ... solve_scale(x, y) 부분 ...
        # (원 코드에서 필요하면 추가)

    print("=== All Done ===")
