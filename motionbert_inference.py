# motionbert_inference.py

import torch
from lib.utils.tools import get_config
from lib.model.model_mesh import MeshRegressor
from lib.model.model import load_backbone
from lib.data.dataset_wild import WildDetDataset
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import flip_thetas_batch
from lib.utils.utils_data import flip_data
from torch.utils.data import DataLoader

def motionbert_inference(vid_path, json_path, config_path, checkpoint_path, device='cuda'):
    # 설정 로드
    args = get_config(config_path)

    # SMPL 모델 초기화
    smpl = SMPL(args.data_root, batch_size=1).to(device)

    # 모델 로드
    model_backbone = load_backbone(args)
    model = MeshRegressor(args, backbone=model_backbone)
    model = model.to(device)
    model.eval()

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)

    # 데이터셋 생성
    dataset = WildDetDataset(json_path, clip_len=args.clip_len)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    smpl_outputs = []

    # 모델 추론
    with torch.no_grad():
        for batch_input in data_loader:
            batch_input = batch_input.to(device).float()
            output = model(batch_input)

            # 플립된 입력에 대한 추론
            batch_input_flip = flip_data(batch_input)
            output_flip = model(batch_input_flip)

            # 결과 평균화
            theta = (output[0]['theta'] + output_flip[0]['theta']) / 2.0
            theta = theta.reshape(-1, 82)  # 72 pose parameters + 10 shape parameters

            # SMPL 파라미터 추출
            pose = theta[:, :72]
            betas = theta[:, 72:]

            smpl_output = smpl(
                betas=betas,
                body_pose=pose[:, 3:],         # Body pose (excluding global orientation)
                global_orient=pose[:, :3],     # Global orientation
                pose2rot=True
            )
            smpl_outputs.append(smpl_output)

    return smpl_outputs