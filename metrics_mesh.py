from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr,get_psnr
from argparse import ArgumentParser
import cv2

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if len(fname.split('.')[0]) <= 5:
            render = Image.open(renders_dir +'/'+ fname)
            if len(fname.split('.')[0])<5:
                fname = fname.split('.')[0].zfill(5)+'.png'
            gt = Image.open(gt_dir +'/'+ fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    print(image_names)
    return renders, gts, image_names




def metrics(render_path,gt_path,savepath,name):
    imagelist = os.listdir(gt_path)
    result = {}
    #renders, gts, image_names = readImages(render_path, gt_path)

    renders, gts, image_names = readImages(render_path, gt_path)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(get_psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    result.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
        
    with open(savepath + f"/{name}_results.json", 'w') as fp:
        json.dump(result, fp, indent=True)
        


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_model_path = project_root / "outputs" / "jumpingjacks_node"

    parser = ArgumentParser(description="Evaluate rendered mesh images")
    parser.add_argument(
        "--model_path",
        default=str(default_model_path),
        help="Model output directory, e.g. outputs/jumpingjacks_node",
    )
    parser.add_argument(
        "--gt_path",
        default=None,
        help="Ground-truth image directory; defaults to test/ours_*/gt_w",
    )
    parser.add_argument(
        "--render_path",
        default=None,
        help="Rendered mesh image directory; defaults to model_path/mesh_image",
    )
    parser.add_argument(
        "--savepath",
        default=None,
        help="Directory for the result JSON; defaults to model_path",
    )
    parser.add_argument("--name", default="mesh_render")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    render_path = Path(args.render_path).expanduser() if args.render_path else model_path / "mesh_image"
    savepath = Path(args.savepath).expanduser() if args.savepath else model_path

    if args.gt_path:
        gt_path = Path(args.gt_path).expanduser()
    else:
        test_dirs = sorted((model_path / "test").glob("ours_*/gt_w"))
        if not test_dirs:
            raise FileNotFoundError(
                f"No white-background GT found under {model_path / 'test'}. "
                "Re-run render_mesh.py once to generate test/ours_*/gt_w, "
                "or pass --gt_path explicitly."
            )
        gt_path = test_dirs[-1]

    for path in (gt_path, render_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {path}")
    savepath.mkdir(parents=True, exist_ok=True)

    metrics(str(render_path), str(gt_path), str(savepath), args.name)
