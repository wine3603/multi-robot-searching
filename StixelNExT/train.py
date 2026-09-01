import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
import os
import numpy as np
import shutil
import cv2
import glob
from losses.stixel_loss import StixelLoss
from models.ConvNeXt import ConvNeXt
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter, draw_heatmap, draw_stixels_on_image
from utilities.visualization import create_composite_image
from dataloader.stixel_multicut import target_transform_gaussian_blur as target_transform
# Optional wandb
if config['logging']['activate']:
    import wandb
from dataloader.stixel_multicut import feature_transform_resize as feature_transform

if config['dataset'] == "kitti":
    from dataloader.stixel_multicut import MultiCutStixelData
    config['grid_step'] = 4
    config['img_height'] = 376
    config['img_width'] = 1248
elif config['dataset'] == "ground":
    from dataloader.ground_data import GroundDataStixel as MultiCutStixelData
    config['grid_step'] = 4
    config['img_height'] = 376
    config['img_width'] = 1248
else:
    from dataloader.stixel_multicut import MultiCutStixelData
    feature_transform = None
    config['grid_step'] = 8
    config['img_height'] = 1200
    config['img_width'] = 1920

# 0.2 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
overall_start_time = datetime.now()


def run_test_inference(model, test_dir, device, epoch, best_iou, best_test_error):
    """Run inference on test directory - following 14_infer_ground_e.py logic."""
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    # Find all original images (exclude already annotated files)
    exts = ["*.jpg", "*.jpeg", "*.png"]
    test_images = []
    for ext in exts:
        for f in glob.glob(os.path.join(test_dir, ext)):
            base = os.path.basename(f)
            # Skip annotated and mask files
            if "_annotated" not in base and "_mask" not in base:
                test_images.append(f)

    if not test_images:
        print(f"No test images found in {test_dir}")
        return

    model.eval()
    print(f"\n🎯 Running inference on {len(test_images)} test images...")

    for img_path in test_images:
        try:
            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]

            # Preprocess: BGR -> RGB -> Tensor (0-255 range) - exactly like 14_infer_ground_e.py
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            # Run inference with resize to STIXEL input size (376x1248)
            with torch.no_grad():
                img_tensor_stixel = torch.nn.functional.interpolate(
                    img_tensor, size=(config['img_height'], config['img_width']),
                    mode='bilinear', align_corners=False
                )
                output = model(img_tensor_stixel)

                # Get channel 0 (occupancy/ground) - model output already has Sigmoid
                # Output shape can be [2, H, W] or [1, 2, H, W]
                if output.dim() == 4:
                    prob_stixel = output[0, 0, :, :].cpu().numpy()
                else:
                    prob_stixel = output[0, :, :].cpu().numpy()

            # Resize back to original image size
            prob_full = cv2.resize(prob_stixel, (w, h), interpolation=cv2.INTER_LINEAR)

            # Binary mask with threshold 0.5
            binary_mask = (prob_full > 0.5).astype(np.uint8) * 255

            # Create overlay visualization (green semi-transparent)
            vis = img_bgr.copy()
            overlay = np.zeros_like(vis)
            overlay[binary_mask > 127] = [0, 200, 0]  # BGR green
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Save with fixed names (overwrite each time)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(test_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(output_path, vis)

            # Also save binary mask
            mask_path = os.path.join(test_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, binary_mask)

            ground_ratio = np.sum(binary_mask > 127) / binary_mask.size
            print(f"  ✓ {base_name}: ground={ground_ratio:.1%}")

        except Exception as e:
            print(f"  ✗ Error processing {img_path}: {e}")

    # Save model performance info to test directory
    perf_path = os.path.join(test_dir, "model_performance.txt")
    with open(perf_path, "w") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Best IoU: {best_iou:.6f}\n")
        f.write(f"Test Error: {best_test_error:.6f}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"  💾 Model performance saved to {perf_path}")

    print(f"✅ Inference complete, saved to {test_dir}")


def main():
    test_dataloader = None
    # Load data
    if config['dataset'] == "ground":
        # ground_data is already at data_path
        dataset_dir = config['data_path']
    else:
        dataset_dir = os.path.join(config['data_path'], config['dataset'])
    # Training data
    training_data = MultiCutStixelData(data_dir=dataset_dir,
                                       phase='training',
                                       transform=feature_transform,
                                       target_transform=target_transform)               # target_transform_gaussian_blur
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'],
                                  num_workers=config['resources']['train_worker'], pin_memory=True, drop_last=True)
    # Validation data
    validation_data = MultiCutStixelData(data_dir=dataset_dir,
                                         phase='validation',
                                         transform=feature_transform,
                                         target_transform=target_transform)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['val_worker'], pin_memory=True, shuffle=False, drop_last=True)

    # Testing data
    if config['explore_data'] or config['test_loss']:
        testing_data = MultiCutStixelData(data_dir=dataset_dir,
                                          phase='testing',
                                          transform=feature_transform,
                                          target_transform=target_transform,
                                          return_original_image=True)
        test_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                     num_workers=config['resources']['test_worker'], pin_memory=True, shuffle=True,
                                     drop_last=True)

    # Define Model
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     target_height=int(training_data.img_size['height'] / config['grid_step']),
                     target_width=int(training_data.img_size['width'] / config['grid_step']),
                     out_channels=2).to(device)

    # Load Weights
    best_model_path = os.path.join("best_model_weights", f"StixelNExT_{config['dataset']}_best.pth")
    # Auto-resume from best model if exists
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(f=best_model_path,
                       map_location=torch.device(device)))
        print(f'✅ Resuming from best model: {best_model_path}')
    elif config['load_weights']:
        weights_file = config['weights_file']
        checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
        run = checkpoint.split('_')[1]
        model.load_state_dict(
            torch.load(f=os.path.join("saved_models", run, weights_file),
                       map_location=torch.device(device)))
        print(f'Weights loaded from: {weights_file}')
    # Loss function
    loss_fn = StixelLoss(alpha=config['loss']['alpha'],
                         beta=config['loss']['beta'],
                         gamma=config['loss']['gamma'])

    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Logger
    if config['logging']['activate']:
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "loss_function": type(loss_fn).__name__,
                                      "loss_alpha": config['loss']['alpha'],
                                      "loss_beta": config['loss']['beta'],
                                      "loss_gamma": config['loss']['gamma'],
                                      "architecture": type(model).__name__,
                                      "dataset": training_data.name,
                                      "epochs": config['num_epochs'],
                                  },
                                  tags=["training"]
                                  )
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    # Explore data
    if config['explore_data']:
        result_interpreter = StixelNExTInterpreter()
        # Ground Truth
        indx = np.random.randint(0, len(testing_data))
        print(indx)
        test_features, test_labels, image = testing_data[indx]
        gt_occ_hm = draw_heatmap(image, test_labels, mtx_map='occ')
        gt_cut_hm = draw_heatmap(image, test_labels, mtx_map='cut')
        gt_stixel = result_interpreter.extract_stixel_from_prediction(test_labels)
        gt_stixel_img = draw_stixels_on_image(image, gt_stixel, color=[0, 255, 0])
        # Prediction
        if config['load_weights']:
            sample = test_features.unsqueeze(0).to(device)
            output = model(sample)
            output = output.cpu().detach()
            output = output.squeeze()
            pred_occ_hm = draw_heatmap(image, output, mtx_map='occ')
            pred_cut_hm = draw_heatmap(image, output, mtx_map='cut')
            thres = 0.48
            pred_stixel = result_interpreter.extract_stixel_from_prediction(output, detection_threshold=thres)
            pred_stixel_img = draw_stixels_on_image(image, pred_stixel, color=[48, 213, 200])
            composite = create_composite_image([gt_occ_hm, pred_occ_hm, gt_cut_hm, pred_cut_hm, gt_stixel_img, pred_stixel_img])
            comment = ""
            composite.save(f"results/{config['weights_file']}_loss-{config['loss']['alpha']}-{config['loss']['beta']}-{config['loss']['gamma']}-{config['loss']['delta']}_{comment}.png")

    # Inspect model
    if config['inspect_model']:
        height = int(config["img_height"])
        width = int(config["img_width"])
        summary(model, (3, height, width))
        if not config['test_loss']:
            print("Running on " + device)

    # testing loss
    if config['test_loss']:
        test_features, test_labels, image = next(iter(test_dataloader))
        data = test_features.to(device)
        output = model(data)
        target = test_labels.to(device)
        print("Input shape: " + str(data.shape))
        print("Output shape: " + str(output.shape))
        print("Running on " + device)
        print("----------------------------------------------------------------")
        print(loss_fn(output, target))

    # Training
    if config['training']:
        from engine import EarlyStopping
        # Early stopping based on validation IoU (higher is better)
        early_stopping = EarlyStopping(tolerance=config['early_stopping']['tolerance'],
                                       min_delta=config['early_stopping']['min_delta'])
        best_test_error = float('inf')
        best_iou = -float('inf')
        best_model_path = os.path.join("best_model_weights", f"StixelNExT_{config['dataset']}_best.pth")
        os.makedirs("best_model_weights", exist_ok=True)

        for epoch in range(config['num_epochs']):
            print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
            train_error = train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                                          device=device, writer=wandb_logger)
            test_error, avg_iou = evaluate(val_dataloader, model, loss_fn,
                                  device=device, epoch=epoch, writer=wandb_logger)

            # Only save if this is the best so far (higher IoU is better)
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_test_error = test_error
                # Save best model (overwrite previous best)
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ New best model saved! Mean IoU = {best_iou:>8f} → {best_model_path}")

                # Copy best model to comparison folder automatically
                comparison_folder = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/"
                if os.path.exists(comparison_folder):
                    dest_compare = os.path.join(comparison_folder, f"StixelNExT_ground_best.pth")
                    shutil.copy(best_model_path, dest_compare)
                    print(f"✅ Best model copied to comparison folder: {dest_compare}")

                # Run inference on test images (pass updated best_iou and best_test_error)
                test_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/test"
                run_test_inference(model, test_dir, device, epoch + 1, best_iou, best_test_error)
            else:
                print(f"Current IoU {avg_iou:>8f} not better than best {best_iou:>8f}, skip saving")

            step_time = datetime.now() - overall_start_time
            print("Time elapsed: {}".format(step_time))

            # early stopping - check based on negative IoU (since early stopping expects lower is better)
            early_stopping.check_stop(-avg_iou)  # negative because lower negative = higher IoU
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}, best Mean IoU = {best_iou:>8f}")
                break

        overall_time = datetime.now() - overall_start_time
        print(f"\nFinished training in {str(overall_time).split('.')[0]}")
        print(f"Best validation Mean IoU: {best_iou:>8f}")
        print(f"Best validation error: {best_test_error:>8f}")
        print(f"Best model: {best_model_path}")
        print(f"Already copied to: /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/StixelNExT_ground_best.pth")
    if config['logging']['activate']:
        wandb.finish()


if __name__ == '__main__':
    main()
