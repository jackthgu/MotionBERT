import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_wild import WildDetDataset
from lib.model.model_mesh import MeshRegressor
from lib.utils.vismo import render_and_save
from lib.utils.utils_data import flip_data
from lib.utils.utils_mesh import flip_thetas_batch
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares
import pdb

class MotionBERTInferencer:
    def __init__(self, config_path, evaluate_path, json_path, vid_path, out_path, **kwargs):
        self.config_path = config_path
        self.evaluate_path = evaluate_path
        self.json_path = json_path
        self.vid_path = vid_path
        self.out_path = out_path
        
        self.pixel = kwargs.get("pixel", False)
        self.focus = kwargs.get("focus", None)
        self.clip_len = kwargs.get("clip_len", 243)
        self.ref_3d_motion_path = kwargs.get("ref_3d_motion_path", None)
        
        self.args = get_config(self.config_path)
        from MotionBERT.lib.utils import utils_smpl
        
        patched_smpl_dir = os.path.join(os.path.dirname(os.path.abspath(self.config_path)), "../../data/mesh")
        patched_smpl_dir = os.path.abspath(patched_smpl_dir)
        print("Updated utils_smpl.SMPL_MODEL_DIR to:", patched_smpl_dir)
        utils_smpl.SMPL_MODEL_DIR = patched_smpl_dir
        
        
        if not os.path.isabs(self.args.data_root):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(self.config_path), '..', '..'))
            self.args.data_root = os.path.join(base_dir, self.args.data_root)
            print("Updated args.data_root:", self.args.data_root)
        
        os.makedirs(self.out_path, exist_ok=True)
        
        self.smpl = SMPL(self.args.data_root, batch_size=1).cuda()
        self.J_regressor = self.smpl.J_regressor_h36m
        
        self._init_model()
        
        self.vid = imageio.get_reader(self.vid_path,  'ffmpeg')
        self.fps_in = self.vid.get_meta_data()['fps']
        self.vid_size = self.vid.get_meta_data()['size']
        
        self._init_dataset()
        
    def _init_model(self):
        start_time = time.time()
        self.backbone = load_backbone(self.args)
        print(f'Init backbone time: {(time.time() - start_time):.2f}s')
        
        self.model = MeshRegressor(self.args, backbone=self.backbone, 
                                   dim_rep=self.args.dim_rep, 
                                   hidden_dim=self.args.hidden_dim, 
                                   dropout_ratio=self.args.dropout)
        
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
            
        print('Loading checkpoint', self.evaluate_path)
        checkpoint = torch.load(self.evaluate_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        
    def _init_dataset(self):
        self.testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
        }
        
        if self.pixel:
            self.dataset = WildDetDataset(self.json_path, clip_len=self.clip_len, 
                                          vid_size=self.vid_size, scale_range=None, focus=self.focus)
        else:
            self.dataset = WildDetDataset(self.json_path, clip_len=self.clip_len, 
                                          scale_range=[1, 1], focus=self.focus)
        self.test_loader = DataLoader(self.dataset, **self.testloader_params)
        
    def _solve_scale(self, x, y):
        """
        카메라 스케일 추정을 위해 least-squares 문제를 풉니다.
        """
        def err(p, x, y):
            return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()
        
        print('Estimating camera transformation.')
        best_res = 1e5
        best_scale = None
        for init_scale in tqdm(range(0, 2000, 5)):
            p0 = [init_scale, 0.0, 0.0, 0.0]
            est = least_squares(err, p0, args=(x.reshape(-1, 3), y.reshape(-1, 3)))
            # est['fun']가 1개의 요소를 가진 numpy.ndarray인 경우, .item()을 통해 스칼라로 변환합니다.
            current_error = est['fun'].item() if isinstance(est['fun'], np.ndarray) else est['fun']
            if current_error < best_res:
                best_res = current_error
                best_scale = est['x'][0]
        print(f'Pose matching error = {float(best_res):.2f} mm.')
        return best_scale
    
    def _export_smpl_parameters(self, smpl_output, output_path):
        vertices = smpl_output.vertices.detach().cpu().numpy()
        joints = smpl_output.joints.detach().cpu().numpy()
        
        body_pose = smpl_output.body_pose.detach().cpu().numpy()
        global_orient = smpl_output.global_orient.detach().cpu().numpy()
        betas = smpl_output.betas.detach().cpu().numpy()
        
        faces = self.smpl.faces
        
        kintree_table = np.array([
            [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 12, 13, 14],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        ], dtype=np.int32)
        
        J = np.matmul(self.J_regressor.cpu().numpy(), vertices[0])
        
        smpl_data = {
            'kintree_table': kintree_table,
            'J': joints,
            'betas': betas[0],
            'v_template': vertices[0],
            'weights': np.zeros((vertices.shape[1], 17)),
            'shapedirs': np.zeros((vertices.shape[1], 3, 10)),
            'J_regressor': self.J_regressor.cpu().numpy(),
            'poses': np.concatenate([global_orient, body_pose], axis=-1),
            'f': faces
        }
        
        np.savez(output_path, **smpl_data)
        print(f"SMPL parameters exported to {output_path}")
    
    def run_inference(self):
        verts_all = []
        reg3d_all = []
        
        with torch.no_grad():
            for batch_input in tqdm(self.test_loader):
                batch_size, clip_frames = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda().float()
                
                output = self.model(batch_input)
                
                batch_input_flip = flip_data(batch_input)
                output_flip = self.model(batch_input_flip)
                output_flip_pose = output_flip[0]['theta'][:, :, :72]
                output_flip_shape = output_flip[0]['theta'][:, :, 72:]
                output_flip_pose = flip_thetas_batch(output_flip_pose)
                output_flip_pose = output_flip_pose.reshape(-1, 72)
                output_flip_shape = output_flip_shape.reshape(-1, 10)
                
                output_flip_smpl = self.smpl(
                    betas=output_flip_shape,
                    body_pose=output_flip_pose[:, 3:],
                    global_orient=output_flip_pose[:, :3],
                    pose2rot=True
                )
                
                output_flip_verts = output_flip_smpl.vertices.detach()
                J_regressor_batch = self.J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
                output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)
                
                output_flip_back = [{
                    'verts': output_flip_verts.reshape(batch_size, clip_frames, -1, 3) * 1000.0,
                    'kp_3d': output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3),
                }]
                output_final = [{}]
                for k, v in output_flip_back[0].items():
                    output_final[0][k] = (output[0][k] + output_flip_back[0][k]) / 2.0
                output = output_final
                
                verts_all.append(output[0]['verts'].cpu().numpy())
                reg3d_all.append(output[0]['kp_3d'].cpu().numpy())
                
                export_path = osp.join(self.out_path, 'my_smpl_params.npz')
                self._export_smpl_parameters(output_flip_smpl, export_path)
                
        verts_all = np.hstack(verts_all)
        verts_all = np.concatenate(verts_all)
        reg3d_all = np.hstack(reg3d_all)
        reg3d_all = np.concatenate(reg3d_all)
        
        if self.ref_3d_motion_path:
            ref_pose = np.load(self.ref_3d_motion_path)
            x = ref_pose - ref_pose[:, :1]
            y = reg3d_all - reg3d_all[:, :1]
            scale = self._solve_scale(x, y)
            root_cam = ref_pose[:, :1] * scale
            verts_all = verts_all - reg3d_all[:, :1] + root_cam
        
        out_video_path = osp.join(self.out_path, 'mesh.mp4')
        render_and_save(verts_all, out_video_path, keep_imgs=False, fps=self.fps_in, draw_face=True)
        print(f"Mesh video saved at {out_video_path}")
        
        return verts_all, reg3d_all
    
    def run(self):
        self.run_inference()

if __name__ == "__main__":
    config_path = "configs/mesh/MB_ft_pw3d.yaml" 
    evaluate_path = "checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin" 
    json_path = "alphapose-results.json" 
    vid_path = "videoplayback.mp4" 
    out_path = "test_mesh" 
    ref_3d_motion_path = "test/X3D.npy" 

    pixel = False  
    focus = None  
    clip_len = 243 
    
    inferencer = MotionBERTInferencer(
        config_path=config_path,
        evaluate_path=evaluate_path,
        json_path=json_path,
        vid_path=vid_path,
        out_path=out_path,
        pixel=pixel,
        focus=focus,
        clip_len=clip_len,
        ref_3d_motion_path=ref_3d_motion_path
    )
    inferencer.run()
    pdb.set_trace()