from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def getModel():
    # load models
    detector_run_id = 'detector-bop-tless-pbr--873074'
    coarse_run_id = 'coarse-bop-tless-pbr--506801'
    refiner_run_id = 'refiner-bop-tless-pbr--233420'
    detector = load_detector(detector_run_id)
    pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id, n_workers=4)
    return detector, pose_predictor


def inference(detector, pose_predictor, image, camera_k):
    # [1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    # [1,3,3]
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    # 2D detector
    # print("start detect object.")
    box_detections = detector.get_detections(images=images, one_instance_per_class=False,
                                             detection_th=0.8, output_masks=False, mask_th=0.9)
    # pose esitimition
    if len(box_detections) == 0:
        return None
    # print("start estimate pose.")
    final_preds, all_preds = pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                                                            n_coarse_iterations=1, n_refiner_iterations=4)
    # print("inference successfully.")
    # result: this_batch_detections, final_preds
    return final_preds.cpu()


def main():
    detector, pose_predictor = getModel()
    print("start...........................................")
    # the target image
    path = "/home/ubuntu/project/cosypose/test.png"
    img = Image.open(path)
    img = np.array(img)
    camera_k = np.array([[585.75607, 0, 320.5], \
                         [0, 585.75607, 240.5], \
                         [0, 0, 1, ]])
    # predict
    pred = inference(detector, pose_predictor, img, camera_k)
    # poses,poses_input,K_crop,boxes_rend,boxes_crop
    print("num of pred:", len(pred))
    for i in range(len(pred)):
        print("object ", i, ":", pred.infos.iloc[i].label, "------\n  pose:", pred.poses[i].numpy(),
              "\n  detection score:", pred.infos.iloc[i].score)


if __name__ == '__main__':
    main()
