import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.my_load_data import my_load_data

from torch_efficient_distloss import flatten_eff_distloss


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--json_dir', required=True,
                        help='path to the transform_test.json. (e.g., */*/transform_test.json)')    
    parser.add_argument('--output_dir', required=True,
                        help='path of the folder to put output image')
    parser.add_argument("--seed", type=int, default=1211,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_val", action='store_true')
    parser.add_argument("--render_test", action='store_true')
    
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg, json_dir):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = my_load_data(cfg.data, json_dir)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    json_dir = args.json_dir
    sub = json_dir.split("/")[-1]
    datadir = json_dir.replace(sub,'')
    cfg.data.datadir = datadir
    print("args",args)
    print("cfg",cfg)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    seed_everything()

    # # load images / poses / camera settings / data split
    # # change to splits=['test']
    # splits = ['val']
    # # if args.render_train:
    # #     splits.append('train')
    # # if args.render_val:
    # #     splits.append('val')
    # # if args.render_test:
    # #     splits.append('test')
    # data_dict = load_everything(args=args, cfg=cfg, splits=splits)

    data_dict = load_everything(args=args, cfg=cfg, json_dir=json_dir)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # load model for rendering
    if args.render_test or args.render_val or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        print("stepsize 123", stepsize)
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # Eval
    if args.render_val: 
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_val_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        print('==============Render Eval==============')
        # print(len(data_dict['poses']))
        # print(len(data_dict['HW']))
        # print(len(data_dict['Ks']))
        # print(len(data_dict['poses'][data_dict['i_val']]))
        # print(len(data_dict['HW'][data_dict['i_val']]))
        # print(len(data_dict['Ks'][data_dict['i_val']]))
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'],
                HW=data_dict['HW'],
                Ks=data_dict['Ks'],
                gt_imgs=None, # test has no gt img.
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)
        
        # Save image here.
        os.makedirs(args.output_dir, exist_ok=True)

        # read json filenames 
        with open(json_dir) as f:
            data = json.load(f)
        filename = []
        for frame in data["frames"]:
            file_path = frame['file_path']
            filename.append(file_path.split("/")[-1])
        
        for i in trange(len(rgbs)):
            # rgb8 = utils.to8b(rgbs[i])
            rgb = rgbs[i]
            # TODO: change filename according to transform_test.py
            png_name = os.path.join(args.output_dir, f'{filename[i]}.png')
            # print(png_name)
            imageio.imwrite(png_name, rgb)
        print(f"All files are dumped into {args.output_dir} successfully.")

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
    
    print('Done')

