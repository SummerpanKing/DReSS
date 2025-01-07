import time
import numpy as np
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import os
import cv2


def normalize_image(image):
    # 获取特征图的最小值和最大值
    min_val = np.min(image, axis=(0, 1))
    max_val = np.max(image, axis=(0, 1))
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, query_bev, reference, ids, positions in bar:

        if scaler:
            with autocast():

                # data (batches) to device   
                query = query.to(train_config.device)
                query_bev = query_bev.to(train_config.device)
                reference = reference.to(train_config.device)

                # -- note that we remove the image dropout and rotation processes to guarantee the positions.
                positions = torch.cat((positions[0].unsqueeze(0), positions[1].unsqueeze(0)), 0).permute(1, 0).to(
                    'cuda')

                # -- visualization
                # import cv2
                # import os
                # for i in range(len(query)):
                #     im_id = i
                #     x1_vis = query[im_id].permute(1, 2, 0).detach().cpu().numpy()
                #     x1_bev = query_bev[im_id].permute(1, 2, 0).detach().cpu().numpy()
                #     y1_vis = reference[im_id].permute(1, 2, 0).detach().cpu().numpy()
                #
                #     x1_im = 255 * normalize_image(x1_vis)
                #     x1_bev = 255 * normalize_image(x1_bev)
                #     y1_im = 255 * normalize_image(y1_vis)
                #
                #     # cv2.imwrite(f"./workspace/{im_id}_x1_vis.jpg", x1_im)
                #     cv2.imwrite(f"./workspace/{im_id}_bev_vis.jpg", x1_bev)
                #     cv2.imwrite(fr"./workspace/{im_id}_y1_vis.jpg", y1_im)
                #
                #     y1_vis_tag = cv2.imread(fr"./workspace/{im_id}_y1_vis.jpg")
                #
                #     # draw gt
                #     y_np = positions.detach().cpu().numpy()
                #     pos = [int(y_np[im_id][0]), int(y_np[im_id][1])]
                #     tp1, tp2 = int(y_np[im_id][0]), int(y_np[im_id][1])
                #     y1_vis_tag = cv2.circle(y1_vis_tag, [int(y_np[im_id][0]), int(y_np[im_id][1])], 5, (0, 0, 255), -1)
                #     cv2.imwrite(fr"./workspace/{im_id}_y1_vis_tag.jpg", y1_vis_tag)
                #
                #     os.remove(fr"./workspace/{im_id}_y1_vis.jpg")

                # Forward pass
                output1, output1_bev, output2, pos_loss = model(x1=query, x2=query_bev, x3=reference,
                                                                positions=positions)
                global_feature_1, global_feature_1_bev, global_feature_2 = output1[0], output1_bev[0], output2[
                    0]  # -- (1024, 1)

                # --
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss1 = loss_function(global_feature_1, global_feature_2, model.module.logit_scale.exp())
                    loss2 = loss_function(global_feature_1, global_feature_1_bev, model.module.logit_scale2.exp())
                    loss3 = loss_function(global_feature_1_bev, global_feature_2, model.module.logit_scale3.exp())
                    loss4 = pos_loss.mean()

                    loss = loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.05 * loss4

                else:
                    loss1 = loss_function(global_feature_1, global_feature_2, model.logit_scale.exp())
                    loss2 = loss_function(global_feature_1, global_feature_1_bev, model.logit_scale2.exp())
                    loss3 = loss_function(global_feature_1_bev, global_feature_2, model.logit_scale3.exp())
                    loss4 = pos_loss

                    loss = loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.05 * loss4

                loss_all = loss
                losses.update(loss_all.item())

            scaler.scale(loss_all).backward()

            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {"loss1": "{:.4f}".format(loss1.item()),
                       "loss2": "{:.4f}".format(0.1 * loss2.item()),
                       "loss3": "{:.4f}".format(0.1 * loss3.item()),
                       "loss4": "{:.4f}".format(0.05 * loss4.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, model_path):
    batch_save_path = rf"{model_path}/img_features_batches"

    if not os.path.exists(batch_save_path):
        os.makedirs(batch_save_path)

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    # -- draw visualization for ablation study
    draw_vis = False
    if draw_vis:
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 3)
        for i in range(iterations):
            bev_ori = cv2.imread(rf"{pic_path}/{i}_bev.jpg")
            pano_ori = cv2.imread(rf"{pic_path}/{i}_pano.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.png")

            bev_shape = bev_ori.shape[:-1]
            pano_shape = pano_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            bev = cv2.resize(bev_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            pano = cv2.resize(pano_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            bev = torch.tensor(bev).permute(2, 0, 1)  # 通道顺序变换
            pano = torch.tensor(pano).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            bev = normalize(bev)[None, :, :, :]
            pano = normalize(pano)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    bev = bev.to(train_config.device)
                    pano = pano.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_bev = model(bev)[1]
                    # img_feature_bev = F.normalize(img_feature_bev, dim=1)
                    img_feature_pano = model(pano)[1]
                    # img_feature_pano = F.normalize(img_feature_pano, dim=1)
                    img_feature_sat = model(sat)[1]
                    # img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_bev = img_feature_bev[0].permute(1, 2, 0)
                    heat_map_bev = torch.mean(heat_map_bev, dim=2).detach().cpu().numpy()
                    heat_map_bev = (heat_map_bev - heat_map_bev.min()) / (heat_map_bev.max() - heat_map_bev.min())
                    heat_map_bev = cv2.resize(heat_map_bev, [bev_shape[1], bev_shape[0]])

                    heat_map_pano = img_feature_pano[0].permute(1, 2, 0)
                    heat_map_pano = torch.mean(heat_map_pano, dim=2).detach().cpu().numpy()
                    heat_map_pano = (heat_map_pano - heat_map_pano.min()) / (heat_map_pano.max() - heat_map_pano.min())
                    heat_map_pano = cv2.resize(heat_map_pano, [pano_shape[1], pano_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_bev = cv2.applyColorMap((heat_map_bev * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_pano = cv2.applyColorMap((heat_map_pano * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_bev = cv2.addWeighted(bev_ori, alpha, colored_image_bev, 1 - alpha, 0)
                    blended_image_pano = cv2.addWeighted(pano_ori, alpha, colored_image_pano, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)

                    cv2.imwrite(rf"{pic_path}/{i}_bev_vis.jpg", blended_image_bev)
                    cv2.imwrite(rf"{pic_path}/{i}_pano_vis.jpg", blended_image_pano)
                    cv2.imwrite(rf"{pic_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    ids_list = []
    locs_list = []
    batch_count = 0

    with torch.no_grad():
        for img, ids, locs in bar:
            ids_list.append(ids)
            locs_list.append(torch.cat((locs[0].unsqueeze(1), locs[1].unsqueeze(1)), dim=1))

            with autocast():
                img = img.to(train_config.device)
                img_feature = model(img)[0]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_feature = img_feature.to(torch.float32)

            # Save the current batch to disk
            torch.save(img_feature, os.path.join(batch_save_path, f'batch_{batch_count}.pt'))
            batch_count += 1

        # Combine ids and locs
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        locs_list = torch.cat(locs_list, dim=0)

    if train_config.verbose:
        bar.close()

    # Load and concatenate all batches from disk
    img_features_list = []
    for i in range(batch_count):
        batch_features = torch.load(os.path.join(batch_save_path, f'batch_{i}.pt'))
        img_features_list.append(batch_features)

    img_features = torch.cat(img_features_list, dim=0)

    # Clean up temporary files
    for i in range(batch_count):
        os.remove(os.path.join(batch_save_path, f'batch_{i}.pt'))

    return img_features, ids_list, locs_list


def predict_cvusa(train_config, model, dataloader, model_path):
    batch_save_path = rf"{model_path}/img_features_batches"

    if not os.path.exists(batch_save_path):
        os.makedirs(batch_save_path)

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    # -- draw visualization for ablation study
    draw_vis = False
    if draw_vis:
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 3)
        for i in range(iterations):
            bev_ori = cv2.imread(rf"{pic_path}/{i}_bev.jpg")
            pano_ori = cv2.imread(rf"{pic_path}/{i}_pano.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.png")

            bev_shape = bev_ori.shape[:-1]
            pano_shape = pano_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            bev = cv2.resize(bev_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            pano = cv2.resize(pano_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            bev = torch.tensor(bev).permute(2, 0, 1)  # 通道顺序变换
            pano = torch.tensor(pano).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            bev = normalize(bev)[None, :, :, :]
            pano = normalize(pano)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    bev = bev.to(train_config.device)
                    pano = pano.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_bev = model(bev)[1]
                    img_feature_pano = model(pano)[1]
                    img_feature_sat = model(sat)[1]

                    heat_map_bev = img_feature_bev[0].permute(1, 2, 0)
                    heat_map_bev = torch.mean(heat_map_bev, dim=2).detach().cpu().numpy()
                    heat_map_bev = (heat_map_bev - heat_map_bev.min()) / (heat_map_bev.max() - heat_map_bev.min())
                    heat_map_bev = cv2.resize(heat_map_bev, [bev_shape[1], bev_shape[0]])

                    heat_map_pano = img_feature_pano[0].permute(1, 2, 0)
                    heat_map_pano = torch.mean(heat_map_pano, dim=2).detach().cpu().numpy()
                    heat_map_pano = (heat_map_pano - heat_map_pano.min()) / (heat_map_pano.max() - heat_map_pano.min())
                    heat_map_pano = cv2.resize(heat_map_pano, [pano_shape[1], pano_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_bev = cv2.applyColorMap((heat_map_bev * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_pano = cv2.applyColorMap((heat_map_pano * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_bev = cv2.addWeighted(bev_ori, alpha, colored_image_bev, 1 - alpha, 0)
                    blended_image_pano = cv2.addWeighted(pano_ori, alpha, colored_image_pano, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)

                    cv2.imwrite(rf"{pic_path}/{i}_bev_vis.jpg", blended_image_bev)
                    cv2.imwrite(rf"{pic_path}/{i}_pano_vis.jpg", blended_image_pano)
                    cv2.imwrite(rf"{pic_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    ids_list = []
    batch_count = 0

    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)

            with autocast():
                img = img.to(train_config.device)
                img_feature = model(img)[0]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_feature = img_feature.to(torch.float32)

            # Save the current batch to disk
            torch.save(img_feature, os.path.join(batch_save_path, f'batch_{batch_count}.pt'))
            batch_count += 1

        # Combine ids and locs
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    # Load and concatenate all batches from disk
    img_features_list = []
    for i in range(batch_count):
        batch_features = torch.load(os.path.join(batch_save_path, f'batch_{i}.pt'))
        img_features_list.append(batch_features)

    img_features = torch.cat(img_features_list, dim=0)

    # Clean up temporary files
    for i in range(batch_count):
        os.remove(os.path.join(batch_save_path, f'batch_{i}.pt'))

    return img_features, ids_list
