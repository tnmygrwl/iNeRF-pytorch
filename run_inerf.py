import os, sys
import math
import numpy as np
import torch
import random
import cv2
import imageio

from pyquaternion import Quaternion
from run_nerf_helpers import *
from run_nerf import create_nerf, render
from load_llff import load_llff_data
from load_blender import load_blender_data

from inerf_sampling import get_random_pixels, get_interest_region_pixels
from so3_helpers import screwExp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=0,
                        help='random seed')

    # inerf dir
    parser.add_argument("--logdir", type=str, default='./inerf_logs/', 
                        help='where to make inerf logs')
    
    # nerf dirs
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego', 
                        help='input data directory')

    # nerf options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # parser.add_argument("--lrate_decay", type=int, default=250, 
    #                     help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # inerf options
    parser.add_argument("--N_rand", type=int, default=1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--num_queries", type=int, default=20, 
                        help='number of queries per scene')
    parser.add_argument("--lrate", type=float, default=0.01, 
                        help='learning rate')
    parser.add_argument("--decay_rate", type=float, default=0.8, 
                        help='learning rate decay rate')
    parser.add_argument("--decay_steps", type=float, default=100, 
                        help='learning rate decay in steps')
    parser.add_argument("--num_steps", type=int, default=300, 
                        help='number of optimization steps per query')
    parser.add_argument("--sampling_type", type=str, default='random', 
                        help='ray sampling strategy')
    parser.add_argument("--debug_render", action='store_true', 
                        help='render full image from current pose estimate')

    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    return parser


def sample_from_unit_sphere():
    '''
    Samples a 3D point randomly from a unit sphere
    '''
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(1 - 2 * np.random.rand())
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])

def get_pose_error(pose1, pose2):
    '''
    Computes the relative translation and rotation error for pose1 and pose2
    '''
    pose1_to_pose2 = np.matmul(pose1, np.linalg.inv(pose2))
    rot_error = np.arccos((np.trace(pose1_to_pose2[:3, :3]) - 1.0) / 2.0) * 180.0 / np.pi
    tran_error = np.linalg.norm(pose1_to_pose2[:3, 3])
    return tran_error, rot_error

def pose_estimation():
    '''
    Performs pose estimation for a random query image from the validation/test dataset
    '''

    # Load config file and set seed
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''
    Load dataset
    '''
    # not using render_poses 
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        row = np.array([0, 0, 0, 1]).reshape(1, 1, 4)
        row = np.tile(row, (poses.shape[0], 1, 1))
        poses = np.concatenate((poses, row), axis=1)

        print('Loaded llff', images.shape, poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)

    # Create log dir and dump the config and args file
    logdir = args.logdir
    expname = args.expname
    os.makedirs(os.path.join(logdir, expname, args.sampling_type), exist_ok=True)
    f = os.path.join(logdir, expname, args.sampling_type, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    f = os.path.join(logdir, expname, args.sampling_type, 'config.txt')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())

    '''
    Initialize NeRF and load pretrained weights
    '''
    _, render_kwargs_test, _, _, _ = create_nerf(args)
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_test.update(bds_dict)

    '''
    Randomly select the query images and initialize with a random pose
    '''
    if args.dataset_type == 'llff':
        all_idx = np.array(i_val)
    elif args.dataset_type == 'blender':
        all_idx = np.concatenate((i_val, i_test))
    
    rand_idx = np.random.permutation(all_idx)[:args.num_queries]
    print(args.sampling_type)
    print(rand_idx)
    random_poses = []
    for idx in list(rand_idx):
        T = poses[idx].astype(float) # ground-truth pose
        axis = sample_from_unit_sphere()
        angle = np.random.uniform(-20, 20) * np.pi / 180 # random [-20, 20] degrees
        if args.dataset_type == 'llff':
            offset = 0.1 # random [-0.1, 0.1] meters
        elif args.dataset_type == 'blender':
            offset = 0.2 # random [-0.2, 0.2] meters
        translation = np.array([np.random.uniform(-offset, offset) for _ in range(3)])
        
        # Converting axis, angle to rotation matrix
        quat = Quaternion(axis=axis, angle=angle)
        T_0_rot = quat.transformation_matrix # 4x4 transformation matrix
        T_0_tran = np.identity(4)
        T_0_tran[:3, 3] = translation
        T_0 = T_0_tran @ T_0_rot
        T_0 = T_0 @ T # initializing T_0 in some vicinity of T
        random_poses.append(T_0)
        # print(f"Initialization offset: translation: {translation}, rotation: {angle * 180 / np.pi} degrees around {axis}")
    
    for i, idx in enumerate(list(rand_idx)):

        idx_path = os.path.join(logdir, expname, args.sampling_type, str(idx))
        os.makedirs(idx_path, exist_ok=True)
        # dump query image
        query = images[idx] # query image
        query_cv2 = query * 255
        query_cv2 = cv2.cvtColor(query_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(idx_path + '/query.png', query_cv2)
        query = torch.from_numpy(query).float().to(device)
        T = poses[idx].astype(float) # ground-truth pose
        T_0 = torch.from_numpy(random_poses[i]).float().to(device)
        print(f"Image ID: {idx}")
        print(f"GT Pose: {T}")
        
        '''
        R6 param to SE3 transformation
        '''
        exp_params = torch.normal(mean=torch.zeros(6), std=1e-6 * torch.ones(6)).to(device)
        exp_params.requires_grad = True
        T_0_hat = screwExp(exp_params) @ T_0
        # initial pose error
        tran_err, rot_err = get_pose_error(T_0_hat.detach().cpu().numpy(), T)
        print(f"Initial Pose: tran error: {tran_err}, rot error: {rot_err}")

        '''
        Optimize: render rays from the current camera pose 
        and backprop the loss to optimize exp_params
        '''
        lrate = args.lrate
        optimizer = torch.optim.Adam(params=[exp_params], lr=lrate, betas=(0.9, 0.999))

        if args.sampling_type == 'interest_region':
            coords = get_interest_region_pixels(H, W, query, args.N_rand, save_path=idx_path)

        f = os.path.join(idx_path, 'log.txt')
        error_log = open(f, 'w')
        error_log.write(f"iteration, loss, tran_error, rot_error\n")                            

        for step in range(args.num_steps):
            ## compute current camera pose
            T_i_hat = screwExp(exp_params) @ T_0
            rays_o, rays_d = get_rays(H, W, focal, T_i_hat[:3, :4]) # (H, W, 3), (H, W, 3)

            # prepare batch of rays to render
            if args.sampling_type == 'random':
                select_coords = get_random_pixels(H, W, args.N_rand)
            elif args.sampling_type == 'interest_region':
                select_coords = torch.from_numpy(np.random.permutation(coords)[:args.N_rand]).long()
        
            
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            query_rgb = query[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            '''
            Render to optimize
            '''
            rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                    retraw=True,
                                                    **render_kwargs_test)
            optimizer.zero_grad()
            img_loss = img2mse(rgb, query_rgb)
            loss = img_loss
            loss.backward()
            optimizer.step()
            ###   update learning rate   ###
            # The learning rate at step t is set as follow α_t = α_0 * 0.8^(t/100)
            new_lrate = lrate * (args.decay_rate ** (step / args.decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            '''
            Save Renders and Error logs
            '''
            if step % 10 == 0:
                # save sampling mask
                # sampled_pixels = np.zeros((H, W)).astype("uint8")
                # select_coords = select_coords.cpu().detach().numpy()
                # sampled_pixels[select_coords[:, 0], select_coords[:, 1]] = 255
                # imageio.imwrite(idx_path + '/m_' + str(step) + '.png', sampled_pixels) 

                # save full-render from the current camera pose
                if args.debug_render:
                    with torch.no_grad():
                        rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=T_i_hat[:3, :4], **render_kwargs_test)
                    rgb_cv2 = rgb.cpu().numpy() * 255
                    rgb_cv2 = cv2.cvtColor(rgb_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(idx_path + '/' + str(step) +'.png', rgb_cv2)

            # current camera pose
            T_i_hat = T_i_hat.detach().cpu().numpy()
            # check pose error
            tran_err, rot_err = get_pose_error(T_i_hat, T)
            # print(f"iteration {step}, loss: {loss.cpu().detach().numpy()}, tran error: {tran_err}, rot error: {rot_err}")
            error_log.write(f"{step}, {loss.cpu().detach().numpy()}, {tran_err}, {rot_err}\n")                       
            error_log.flush()
        error_log.close() 

if __name__=='__main__':
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    pose_estimation()