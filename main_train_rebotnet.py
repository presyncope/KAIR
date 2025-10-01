import argparse, os, logging, random, math, sys
from typing import cast
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2

from utils import utils_logger
from utils import utils_option as option
from utils import utils_image as util

from data.select_dataset import define_Dataset
from models.select_model import define_Model

default_json_path = 'options/rebotnet/001_train_rebotnet_videosr_reds.json'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=default_json_path, help='Path to option JSON file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    return opt

def main():
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    opt = parse_args()
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # update opt
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="G",
        pretrained_path=opt["path"]["pretrained_netG"],
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="E",
        pretrained_path=opt["path"]["pretrained_netE"],
    )
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="optimizerG",
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # save opt to  a '../option.json' file
    option.save(opt)

    # return None for missing key
    opt = cast(option.NoneDict, option.dict_to_nonedict(opt))

    # configure logger
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # seed
    seed: int | None = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    print(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    train_loader, test_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))

            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=True,
                pin_memory=True,
            )

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist.")
    if test_loader is None:
        raise ValueError("The test data does not exist.")

    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()
    logger.info(model.info_network())
    logger.info(model.info_params())

    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    for epoch in tqdm(range(1000000), desc="Training Epochs"):
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # 1) update learning rate
            model.update_learning_rate(current_step)

            # 2) feed patch pairs
            model.feed_data(train_data)

            # 3) optimize parameters
            model.optimize_parameters(current_step)

            # 4) training information
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = f'<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                for k, v in logs.items():  # merge log information into message
                    message += f'{k}: {v:.3e} '
                logger.info(message)

            # 5) save model
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # 6) testing
            if current_step % opt['train']['checkpoint_test'] == 0:
                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []
                test_results['psnr_y'] = []
                test_results['ssim_y'] = []

                for idx, test_data in enumerate(test_loader):
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals(need_H=True)
                    output = cast(torch.Tensor, visuals['E'])
                    gt = cast(torch.Tensor, visuals['H'])
                    folder = test_data['folder']

                    test_results_folder = OrderedDict()
                    test_results_folder['psnr'] = []
                    test_results_folder['ssim'] = []
                    test_results_folder['psnr_y'] = []
                    test_results_folder['ssim_y'] = []

                    for i in range(output.shape[0]):
                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        img = output[i, ...].clamp_(0, 1).numpy()
                        if img.ndim == 3:
                            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                        if opt['val']['save_img']:
                            save_dir = opt['path']['images']
                            util.mkdir(save_dir)
                            seq_ = os.path.basename(test_data['lq_path'][i][0]).split('.')[0]
                            os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                            cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png', img)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        img_gt = gt[i, ...].clamp_(0, 1).numpy()
                        if img_gt.ndim == 3:
                            img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                        img_gt = np.squeeze(img_gt)

                        test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                        if img_gt.ndim == 3:  # RGB image
                            img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                            img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                            test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                            test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                        else:
                            test_results_folder['psnr_y'] = test_results_folder['psnr']
                            test_results_folder['ssim_y'] = test_results_folder['ssim']

                    psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                    ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                    psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                    ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

                    if gt is not None:
                        logger.info('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                    'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                    format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)
                        test_results['psnr_y'].append(psnr_y)
                        test_results['ssim_y'].append(ssim_y)
                    else:
                        logger.info('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

                # summarize psnr/ssim
                if gt is not None: #type: ignore[unreachable]
                    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                    logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                        epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

            if current_step > opt['train']['total_iter']:
                logger.info('Finish training.')
                model.save(current_step)
                sys.exit()

if __name__ == '__main__':
    main()
