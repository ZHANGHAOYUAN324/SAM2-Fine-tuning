# -*- coding: utf-8 -*-
"""
Complete HeartChambersSAMTrainer Class for fine-tuning SAM2 on heart chamber CT scans.
Includes data loading, preprocessing, augmentation, multi-class training, validation,
and an ADDED 'inference' method similar in structure to the SAMTrainer example.

NOTE: The 'inference' method assumes 'initialize_model' has been successfully run beforehand.
The config loading issue within 'initialize_model' itself is NOT solved here,
it relies on being called with correct relative paths & CWD during initialization.

FIXED: Corrected NameError in inference method's mask combination loop.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import os
import re
import json
import cv2
import torch
import torch.nn.utils
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hydra # Import hydra for manual config loading
from omegaconf import DictConfig, OmegaConf # For type hinting config

# Make sure the sam2 library is correctly installed and importable
try:
    from sam2.build_sam import build_sam2 # Still used in initialize_model for training setup
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # Import necessary model components for manual instantiation in inference
    # !!! You might need to adjust these imports based on your specific sam2 version/structure !!!
    # from sam2.modeling.sam import Sam # Example: Main model class
    # from sam2.modeling.image_encoder import SamHieraImageEncoder # Example component
    # from sam2.modeling.prompt_encoder import PromptEncoder # Example component
    # from sam2.modeling.mask_decoder import TwoWayTransformerMaskDecoder # Example component

except ImportError:
    print("Error: sam2 library or required model components not found. Please ensure it is installed.")
    raise ImportError("sam2 library or required model components not found.")

class HeartChambersSAMTrainer:
    """
    A class for training and fine-tuning SAM2 on heart chamber CT scans.
    Includes an 'inference' method for prediction using fine-tuned weights.
    """

    def __init__(
        self,
        data_dir: str,
        model_cfg: str, # Should be the RELATIVE path used successfully during training init
        checkpoint_path: str, # Path to PRETRAINED checkpoint for initial loading
        device: str = "cuda"
    ) -> None:
        """
        Initialize the heart chambers SAM trainer.

        Args:
            data_dir: Root directory containing the dataset.
            model_cfg: Path to the SAM model configuration file (relative path recommended).
            checkpoint_path: Path to the *pretrained* model checkpoint for initial loading by build_sam2.
            device: Device to use ('cuda' or 'cpu').
        """
        self.data_dir = data_dir
        self.a2c_dir = os.path.join(data_dir, "a2c")
        self.a3c_dir = os.path.join(data_dir, "a3c")
        self.a4c_dir = os.path.join(data_dir, "a4c")
        self.device = device
        self.model_cfg = model_cfg # Store the relative path
        self.checkpoint_path = checkpoint_path # Store pretrained path
        self.abs_config_dir = None # Will be set later if needed for manual loading

        self.chamber_classes = { "background": 0, "LA": 1, "RA": 2, "LV": 3, "RV": 4, "myocardium": 5 }
        self.class_names = ["background", "LA", "RA", "LV", "RV", "myocardium"]
        self.class_colors = [ [0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0] ]

        self.sam_model = None # Model used for training/inference base
        self.predictor = None # Predictor instance
        self.train_data = []
        self.test_data = []

        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Switching to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Try to determine absolute config directory path early on
        # Assumes CWD is correct (e.g., segment-anything-2) when __init__ is called
        try:
            # Check if model_cfg is relative and construct absolute path to its directory
            if not os.path.isabs(self.model_cfg):
                 potential_cfg_abs_path = os.path.abspath(self.model_cfg)
                 current_dir = os.path.dirname(potential_cfg_abs_path)
                 while current_dir != "/" and os.path.basename(current_dir) != 'configs':
                       current_dir = os.path.dirname(current_dir)
                 if os.path.basename(current_dir) == 'configs':
                      self.abs_config_dir = current_dir
                      print(f"Determined absolute config directory: {self.abs_config_dir}")
            elif os.path.isabs(self.model_cfg): # If model_cfg was absolute
                 config_dir_maybe = os.path.dirname(self.model_cfg)
                 if os.path.isfile(self.model_cfg): config_dir_maybe = os.path.dirname(config_dir_maybe)
                 if os.path.isdir(config_dir_maybe) and "configs" in config_dir_maybe:
                      self.abs_config_dir = config_dir_maybe
                      print(f"Using absolute config directory based on model_cfg path: {self.abs_config_dir}")

            if self.abs_config_dir is None:
                 print("Warning: Could not reliably determine absolute config directory path for manual loading in inference.")
        except Exception as e:
             print(f"Warning: Error determining absolute config directory path: {e}")


    # --- Data Handling Methods (Keep as before) ---
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """ Prepare training and testing datasets by patient ID. (Keep implementation) """
        all_entries = []
        view_dirs = {"a2c": self.a2c_dir, "a3c": self.a3c_dir, "a4c": self.a4c_dir}
        print(f"Searching for data in: {', '.join(view_dirs.values())}")
        for view_type, view_dir in view_dirs.items():
            if not os.path.isdir(view_dir): print(f"Warning: Directory missing: {view_dir}. Skipping."); continue
            print(f"Processing view: {view_type} in {view_dir}"); found_files = 0
            for root, _, files in os.walk(view_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        json_path = os.path.abspath(os.path.join(root, file)); base_name = os.path.splitext(file)[0]
                        image_path_rel = os.path.join(root, f"{base_name}.png"); image_path_abs = os.path.abspath(image_path_rel); image_exists = os.path.isfile(image_path_abs)
                        has_json_image_data = False
                        try:
                            if not os.path.isfile(json_path): continue
                            with open(json_path, 'r') as f_check:
                                if json.load(f_check).get("imageData"): has_json_image_data = True
                        except: pass
                        if not image_exists and not has_json_image_data: continue
                        patient_id = None; match = re.search(r'[Pp]atient[A-Za-z]?(\d+)', base_name, re.IGNORECASE)
                        if match: patient_id = f"P{match.group(1)}"
                        else: patient_id = base_name.split('_')[0] if '_' in base_name else base_name
                        all_entries.append({ "image_path": image_path_abs if image_exists else None, "annotation_path": json_path, "view_type": view_type, "patient_id": patient_id, "has_json_image": has_json_image_data })
                        found_files += 1
            print(f"Found {found_files} JSON files in {view_dir}")
        if not all_entries: raise ValueError(f"No valid JSON annotation files found!")
        patient_groups = {};
        for entry in all_entries: pid = entry["patient_id"]; patient_groups.setdefault(pid, []).append(entry)
        patient_ids = list(patient_groups.keys())
        if len(patient_ids) < 2: raise ValueError(f"Only {len(patient_ids)} unique patient ID(s) found.")
        if not (0 < test_size < 1): raise ValueError("test_size must be between 0 and 1.")
        try: train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=random_state)
        except ValueError as e: print(f"Error during train_test_split: {e}"); split_idx = int(len(patient_ids) * (1 - test_size)); train_ids, test_ids = patient_ids[:split_idx], patient_ids[split_idx:]
        self.train_data = []; self.test_data = []
        for patient_id in train_ids:
             if patient_id in patient_groups: self.train_data.extend(patient_groups[patient_id]) # Check key exists
        for patient_id in test_ids:
             if patient_id in patient_groups: self.test_data.extend(patient_groups[patient_id]) # Check key exists
        if not self.train_data or not self.test_data: print(f"Warning: Train/Test split resulted in empty set(s).")
        print(f"\nData Preparation Summary:"); print(f"Total JSON entries: {len(all_entries)}"); print(f"Total unique patients: {len(patient_ids)}")
        print(f"Training patients: {len(train_ids)}, Testing patients: {len(test_ids)}"); print(f"Training samples: {len(self.train_data)}, Testing samples: {len(self.test_data)}")
        print(f"Training views: {self._count_view_types(self.train_data)}"); print(f"Testing views: {self._count_view_types(self.test_data)}")

    def _count_view_types(self, data: List[Dict]) -> Dict[str, int]:
        """Counts the distribution of view types."""
        view_counts = {};
        for entry in data: view = entry.get('view_type', 'unknown'); view_counts[view] = view_counts.get(view, 0) + 1
        return view_counts

    def _parse_json_annotation(self, json_path: str, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Parses labelme format JSON annotation for multi-class mask."""
        height, width = image_shape; mask = np.zeros((height, width), dtype=np.uint8)
        try:
            if not os.path.isfile(json_path): print(f"Error: JSON file not found at {json_path}"); return None
            with open(json_path, 'r') as f: annotation_data = json.load(f)
        except Exception as e: print(f"Error reading JSON {json_path}: {e}"); return None
        shapes = annotation_data.get('shapes', [])
        if not shapes: return mask
        for shape in shapes:
            label = shape.get('label'); points = shape.get('points', []); shape_type = shape.get('shape_type')
            if not label or not points or shape_type != 'polygon': continue
            class_id = self.chamber_classes.get(label)
            if class_id is None and label.upper() == 'M' and 'myocardium' in self.chamber_classes: class_id = self.chamber_classes['myocardium']
            if class_id is None or class_id == 0: continue
            try:
                points_array = np.array(points, dtype=np.int32)
                if points_array.ndim == 2: points_array = points_array[:, np.newaxis, :]
                cv2.fillPoly(mask, [points_array], color=int(class_id))
            except Exception as e: print(f"Warning: cv2.fillPoly failed for label '{label}' in {json_path}: {e}")
        return mask

    def read_batch( self, data: List[Dict[str, Any]], visualize: bool = False, process_specific_class: Optional[int] = None, apply_augmentation: bool = True ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """ Read and preprocess a random batch. (Keep implementation) """
        if not data: print("Error: read_batch called with empty data."); return None, None, None, 0
        entry_index = np.random.randint(len(data)); entry = data[entry_index]
        json_path = entry["annotation_path"]; image_path = entry.get("image_path"); has_json_image = entry.get("has_json_image", False)
        image = None
        try:
            if has_json_image:
                with open(json_path, 'r') as f: annotation_data = json.load(f)
                image_data_base64 = annotation_data.get("imageData")
                if image_data_base64: image_data = base64.b64decode(image_data_base64); image_np = np.frombuffer(image_data, np.uint8); image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                if image is None and image_data_base64: print(f"Warning: Failed to decode image from JSON: {json_path}")
            if image is None and image_path:
                if os.path.isfile(image_path): image = cv2.imread(image_path)
                if image is None and os.path.isfile(image_path): print(f"Error: Failed to read image file: {image_path}"); return None, None, None, 0
                elif image_path and not os.path.isfile(image_path): print(f"Error: Image file path not found: {image_path}"); return None, None, None, 0
            if image is None: print(f"Error: Could not load image data for {json_path}."); return None, None, None, 0
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); original_shape = image_rgb.shape[:2]
            multi_class_mask = self._parse_json_annotation(json_path, original_shape) # Use self method
            if multi_class_mask is None: print(f"Error: Failed to parse annotations from {json_path}"); return None, None, None, 0
            target_size = 1024; h, w = original_shape; scale = target_size / max(h, w); new_h, new_w = int(h * scale), int(w * scale)
            resized_image = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_multi_class_mask = cv2.resize(multi_class_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if apply_augmentation: resized_image, resized_multi_class_mask = self._apply_augmentation(resized_image, resized_multi_class_mask)
            binary_mask_for_points = np.zeros_like(resized_multi_class_mask, dtype=np.uint8); class_ids_present = []
            if process_specific_class is not None and process_specific_class > 0:
                class_mask = (resized_multi_class_mask == process_specific_class).astype(np.uint8)
                if np.sum(class_mask) > 0: binary_mask_for_points = class_mask; class_ids_present = [process_specific_class]
            else:
                unique_classes = np.unique(resized_multi_class_mask)
                for class_id in unique_classes:
                    if class_id > 0: class_mask = (resized_multi_class_mask == class_id).astype(np.uint8); binary_mask_for_points = np.maximum(binary_mask_for_points, class_mask); class_ids_present.append(int(class_id))
            points = []; num_points_to_sample = len(class_ids_present)
            if num_points_to_sample > 0 and np.sum(binary_mask_for_points) > 0:
                kernel = np.ones((5, 5), np.uint8); eroded_mask = cv2.erode(binary_mask_for_points, kernel, iterations=1)
                coords = np.argwhere(eroded_mask > 0)
                if len(coords) == 0: coords = np.argwhere(binary_mask_for_points > 0)
                if len(coords) > 0:
                    num_available = len(coords); sample_indices = np.random.choice(num_available, num_points_to_sample, replace=(num_points_to_sample > num_available))
                    points = coords[sample_indices][:, ::-1] # Swap columns -> [x, y]
            points = np.array(points, dtype=np.int32)
            if num_points_to_sample > 0 and points.shape[0] != num_points_to_sample: print(f"Warning: Sampled {points.shape[0]}/{num_points_to_sample} points for {json_path}."); return None, None, None, 0
            final_binary_mask = np.expand_dims(binary_mask_for_points, axis=0) # Shape [1, H, W]
            if points.ndim == 2: final_points = np.expand_dims(points, axis=1) # Shape [N, 1, 2]
            elif points.size == 0: final_points = np.empty((0, 1, 2), dtype=np.int32)
            else: print(f"Warning: Unexpected points shape {points.shape}"); final_points = np.empty((0, 1, 2), dtype=np.int32)
            if visualize: self._visualize_batch(resized_image, binary_mask_for_points, points)
            return resized_image, final_binary_mask, final_points, num_points_to_sample
        except Exception as e: print(f"Error processing batch for {entry.get('annotation_path', 'N/A')}: {e}"); import traceback; traceback.print_exc(); return None, None, None, 0

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply data augmentation. (Keep implementation) """
        if np.random.rand() > 0.5: image = cv2.flip(image, 1); mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.8: image = cv2.flip(image, 0); mask = cv2.flip(mask, 0)
        alpha = 1.0 + np.random.uniform(-0.1, 0.1); beta = 0 + np.random.uniform(-10, 10)
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-10, 10); h, w = image.shape[:2]; center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask

    def _visualize_batch( self, image: np.ndarray, mask: np.ndarray, points: np.ndarray ) -> None:
        """ Visualize batch data. (Keep implementation) """
        plt.figure(figsize=(15, 5)); plt.subplot(1, 3, 1); plt.title('Processed Image'); plt.imshow(image); plt.axis('off')
        plt.subplot(1, 3, 2); plt.title('Binary Mask (for Point Sampling)'); plt.imshow(mask, cmap='gray'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.title('Mask with Sampled Points'); plt.imshow(mask, cmap='gray')
        if points.size > 0:
            num_points = points.shape[0]; point_colors = plt.cm.viridis(np.linspace(0, 1, num_points))
            plt.scatter(points[:, 0], points[:, 1], c=point_colors, s=100, marker='o', edgecolor='white', linewidth=1.5)
        plt.axis('off'); plt.tight_layout(); plt.show()

    # --- Model Initialization (Keep as before, relies on build_sam2) ---
    def initialize_model(self) -> None:
        """
        Initialize the SAM model and predictor for TRAINING.
        Uses build_sam2 with the stored relative path. Freezes encoder.
        NOTE: This method might fail if CWD or relative model_cfg path is incorrect.
        """
        print("Initializing SAM model for training...")
        try:
            # Assumes self.model_cfg is a relative path that works when CWD is correct
            self.sam_model = build_sam2(
                self.model_cfg,
                self.checkpoint_path, # Load PRETRAINED weights initially
                device=self.device
            )
            self.predictor = SAM2ImagePredictor(self.sam_model)
            print(f"Model loaded successfully onto {self.device} using build_sam2.")
        except Exception as e: # Catch potential Hydra errors here too
            print(f"Error initializing SAM model using build_sam2: {e}")
            print("Please ensure the notebook's CWD is correct and the relative model_cfg path is valid.")
            import traceback; traceback.print_exc()
            raise # Re-raise to stop if initialization fails

        # --- Freeze Image Encoder ---
        encoder_frozen = False; possible_encoder_names = ['sam_image_encoder', 'image_encoder']
        for name in possible_encoder_names:
            if hasattr(self.predictor.model, name):
                encoder_module = getattr(self.predictor.model, name)
                if encoder_module is not None:
                    print(f"Freezing parameters of image encoder module: '{name}'")
                    for param in encoder_module.parameters(): param.requires_grad = False
                    encoder_frozen = True; break
        if not encoder_frozen: print(f"WARNING: Could not freeze image encoder.")

        # --- Set Decoders/Encoders to Train Mode ---
        trainable_components = []
        possible_decoder_names = ['sam_mask_decoder', 'mask_decoder']
        possible_prompt_enc_names = ['sam_prompt_encoder', 'prompt_encoder']
        for name in possible_decoder_names:
             if hasattr(self.predictor.model, name):
                 module = getattr(self.predictor.model, name)
                 if module is not None: module.train(True); trainable_components.append(name); print(f"Set '{name}' to training mode."); break
        for name in possible_prompt_enc_names:
             if hasattr(self.predictor.model, name):
                 module = getattr(self.predictor.model, name)
                 if module is not None: module.train(True); trainable_components.append(name); print(f"Set '{name}' to training mode."); break
        if not trainable_components: print("WARNING: Could not set decoder/prompt encoder to train mode.")
        num_trainable_params = sum(p.numel() for p in self.predictor.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")
        if num_trainable_params == 0: print("WARNING: No trainable parameters detected.")

    # --- Training Methods (Keep as before) ---
    def train( self, steps: int = 2000, learning_rate: float = 1e-5, weight_decay: float = 2e-4, accumulation_steps: int = 8, scheduler_step_size: int = 300, scheduler_gamma: float = 0.5, checkpoint_interval: int = 500, validation_interval: int = 500, model_name_prefix: str = "./models/heart_chambers_sam" ) -> None:
        """ Train the SAM model using combined binary masks (single point). (Keep implementation) """
        # --- Implementation from previous version ---
        if not self.predictor: raise ValueError("Model not initialized."); print("Call initialize_model() first.")
        if not self.train_data: raise ValueError("Training data not prepared."); print("Call prepare_data() first.")
        trainable_params = [p for p in self.predictor.model.parameters() if p.requires_grad]
        if not trainable_params: print("Warning: No trainable parameters found."); return
        optimizer = AdamW(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scaler = GradScaler(enabled=(self.device == 'cuda')); mean_iou = 0.0; best_val_iou = 0.0
        print(f"\n--- Starting Training (Combined Mask Mode - Single Point Prompt) ---"); print(f"Total steps: {steps}") # ... print other params ...
        optimizer.zero_grad()
        for step in range(1, steps + 1):
            self.predictor.model.train()
            with autocast(device_type=self.device.split(':')[0], enabled=(self.device == 'cuda')):
                image, mask, input_points_all, num_masks_all = self.read_batch(self.train_data, visualize=(step == 1), apply_augmentation=True)
                if (image is None or mask is None or num_masks_all == 0 or not isinstance(input_points_all, np.ndarray) or input_points_all.size == 0): print(f"Step {step}/{steps}: Skipping invalid batch read."); continue
                point_index = np.random.randint(num_masks_all); input_point_single = input_points_all[point_index:point_index+1, :, :]; input_label_single = np.ones((1, 1), dtype=np.int32); num_masks_single = 1
                try: self.predictor.set_image(image)
                except Exception as e: print(f"Step {step}/{steps}: Error setting image: {e}"); continue
                if not isinstance(input_point_single, np.ndarray) or not isinstance(input_label_single, np.ndarray): print(f"Step {step}/{steps}: Invalid type for single point/label."); continue
                try: mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point_single, input_label_single, box=None, mask_logits=None, normalize_coords=True)
                except Exception as e: print(f"Step {step}/{steps}: Error preparing prompts: {e}"); continue
                if (unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0): print(f"Step {step}/{steps}: Invalid prepared prompts."); continue
                try: sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
                except Exception as e: print(f"Step {step}/{steps}: Error generating prompt embeddings: {e}"); continue
                batched_mode = False # Should be False for single point
                if not hasattr(self.predictor, '_features') or "high_res_feats" not in self.predictor._features: print(f"Step {step}/{steps}: Predictor features missing."); continue
                high_res_features = [f[-1].unsqueeze(0) for f in self.predictor._features["high_res_feats"]]
                if "image_embed" not in self.predictor._features: print(f"Step {step}/{steps}: Predictor image embedding missing."); continue
                try: low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder( image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0), image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features )
                except Exception as e: print(f"Step {step}/{steps}: Error during mask decoding: {e}"); continue
                if not hasattr(self.predictor, '_orig_hw') or not self.predictor._orig_hw: print(f"Step {step}/{steps}: Predictor original dims missing."); continue
                prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])
                gt_mask = torch.tensor(mask.astype(np.float32)).to(self.device); prd_mask_logits = prd_masks[:, 0]
                if gt_mask.ndim == 2: gt_mask = gt_mask.unsqueeze(0)
                elif gt_mask.ndim != 3: continue
                if prd_mask_logits.ndim == 2: prd_mask_logits = prd_mask_logits.unsqueeze(0)
                elif prd_mask_logits.ndim != 3: continue
                if gt_mask.shape[0] != prd_mask_logits.shape[0]: print(f"Step {step}/{steps}: Batch dim mismatch: gt={gt_mask.shape[0]}, pred={prd_mask_logits.shape[0]}. Skipping."); continue
                prd_mask = torch.sigmoid(prd_mask_logits)
                if gt_mask.numel() == 0 or gt_mask.shape[1] == 0 or gt_mask.shape[2] == 0: continue
                gt_sum = gt_mask.sum(); gt_area = float(gt_mask.shape[1] * gt_mask.shape[2])
                if gt_area == 0: continue
                pos_ratio = gt_sum / gt_area; pos_weight = torch.clamp(1.0 - pos_ratio, min=0.1, max=0.9)
                seg_loss = (-pos_weight * gt_mask * torch.log(prd_mask + 1e-8) - (1.0 - pos_weight) * (1.0 - gt_mask) * torch.log(1.0 - prd_mask + 1e-8)).mean()
                with torch.no_grad(): gt_mask_bool = gt_mask > 0.5; prd_mask_bool = prd_mask > 0.5; inter = (gt_mask_bool & prd_mask_bool).sum(dim=(1, 2)).float(); union = (gt_mask_bool | prd_mask_bool).sum(dim=(1, 2)).float(); iou = inter / (union + 1e-8)
                if prd_scores.ndim != 2 or prd_scores.shape[0] != iou.shape[0]: score_loss = torch.tensor(0.0, device=self.device)
                else: score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                dice_score = (2.0 * inter) / (gt_mask_bool.sum(dim=(1, 2)).float() + prd_mask_bool.sum(dim=(1, 2)).float() + 1e-8); dice_loss = 1.0 - dice_score.mean()
                if not torch.isfinite(seg_loss): seg_loss = torch.tensor(0.0, device=self.device)
                if not torch.isfinite(score_loss): score_loss = torch.tensor(0.0, device=self.device)
                if not torch.isfinite(dice_loss): dice_loss = torch.tensor(0.0, device=self.device)
                loss = seg_loss * 0.7 + score_loss * 0.2 + dice_loss * 0.1; loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (step % accumulation_steps == 0) or (step == steps):
                scaler.unscale_(optimizer);
                if trainable_params: torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            with torch.no_grad(): current_iou = float(iou.mean().cpu().detach()) if torch.isfinite(iou.mean()) else 0.0
            if step == 1: mean_iou = current_iou
            else: mean_iou = (1.0 - 0.01) * mean_iou + 0.01 * current_iou
            if step % 100 == 0: current_lr = optimizer.param_groups[0]['lr']; log_loss = loss.item() * accumulation_steps; print(f"Step {step}/{steps}: Train IoU (EMA) = {mean_iou:.4f}, Batch IoU = {current_iou:.4f}, Loss = {log_loss:.4f}, LR = {current_lr:.6f}")
            if (step % validation_interval == 0 or step == steps) and self.test_data:
                val_iou = self._validate(num_samples=min(50, len(self.test_data)))
                print(f"--- Step {step}/{steps}: Validation IoU = {val_iou:.4f} ---")
                model_dir = os.path.dirname(model_name_prefix);
                if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                if val_iou > best_val_iou and model_dir and os.path.exists(model_dir):
                    best_val_iou = val_iou; best_model_path = f"{model_name_prefix}_best.pth"
                    try: torch.save(self.predictor.model.state_dict(), best_model_path); print(f"Saved best model to {best_model_path}")
                    except Exception as e: print(f"Error saving best model: {e}")
            if step % checkpoint_interval == 0:
                 model_dir = os.path.dirname(model_name_prefix);
                 if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                 if model_dir and os.path.exists(model_dir):
                    checkpoint_name = f"{model_name_prefix}_{step}.pth";
                    try: torch.save(self.predictor.model.state_dict(), checkpoint_name) # print(f"Saved checkpoint: {checkpoint_name}")
                    except Exception as e: print(f"Error saving checkpoint: {e}")
        print(f"\n--- Training Completed ---"); print(f"Best validation IoU: {best_val_iou:.4f}")
        final_model_path = f"{model_name_prefix}_final.pth"; model_dir = os.path.dirname(model_name_prefix)
        if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
        if model_dir and os.path.exists(model_dir):
            try: torch.save(self.predictor.model.state_dict(), final_model_path); print(f"Final model saved: {final_model_path}")
            except Exception as e: print(f"Error saving final model: {e}")

    def _validate( self, num_samples: int = 20 ) -> float:
        """ Run validation using combined masks. (Keep implementation) """
        if not self.test_data: print("Warning: No test data for validation."); return 0.0
        if not self.predictor: print("Warning: Predictor not initialized."); return 0.0
        self.predictor.model.eval(); total_iou = 0.0; valid_samples_processed = 0; valid_samples_with_iou = 0
        actual_samples_to_run = min(num_samples, len(self.test_data)); indices = range(actual_samples_to_run)
        with torch.no_grad():
            for i in indices:
                entry = self.test_data[i];
                try: image, mask, input_point, num_masks = self.read_batch([entry], visualize=False, apply_augmentation=False)
                except Exception as e: print(f"Validation Error reading batch {i}: {e}"); continue
                if (image is None or mask is None or num_masks == 0 or not isinstance(input_point, np.ndarray) or input_point.size == 0): continue
                valid_samples_processed += 1; input_label = np.ones((num_masks, 1), dtype=np.int32)
                try: self.predictor.set_image(image)
                except Exception as e: print(f"Validation Error setting image {i}: {e}"); continue
                try:
                    point_coords_np = input_point[:, 0, :]; point_labels_np = input_label[:, 0]
                    if not isinstance(point_coords_np, np.ndarray) or not isinstance(point_labels_np, np.ndarray): print(f"Validation Error: Invalid point types {i}."); continue
                    pred_masks_val, scores_val, logits_val = self.predictor.predict(point_coords=point_coords_np, point_labels=point_labels_np, multimask_output=False)
                except Exception as e: print(f"Validation Error during prediction {i}: {e}"); continue
                if pred_masks_val is None or len(pred_masks_val) == 0: continue
                combined_pred_mask = np.zeros_like(pred_masks_val[0], dtype=bool);
                for p_mask in pred_masks_val: combined_pred_mask |= p_mask.astype(bool)
                gt_mask_val = mask[0].astype(bool); intersection = np.logical_and(gt_mask_val, combined_pred_mask).sum(); union = np.logical_or(gt_mask_val, combined_pred_mask).sum()
                if union > 0: iou = intersection / union; total_iou += iou; valid_samples_with_iou += 1
        self.predictor.model.train(); avg_iou = total_iou / max(1, valid_samples_with_iou); return avg_iou

    def train_multiclass( self, steps_per_class: int = 500, learning_rate: float = 1e-5, weight_decay: float = 2e-4, accumulation_steps: int = 8, scheduler_step_size: int = 100, scheduler_gamma: float = 0.5, checkpoint_interval: int = 200, validation_interval: int = 200, model_name_prefix: str = "./models/heart_chambers_sam_multiclass" ) -> None:
        """ Train the SAM model using a class-by-class approach. (Keep implementation) """
        if not self.predictor: raise ValueError("Model not initialized."); print("Call initialize_model() first.")
        if not self.train_data: raise ValueError("Training data not prepared."); print("Call prepare_data() first.")
        classes_to_train = [cid for cid in self.chamber_classes.values() if cid > 0]
        if not classes_to_train: print("Warning: No classes found to train."); return
        print(f"\n--- Starting Training (Multi-Class Sequential Mode) ---"); print(f"Classes: {[self.class_names[i] for i in classes_to_train]}"); print(f"Steps per class: {steps_per_class}")
        trainable_params = [p for p in self.predictor.model.parameters() if p.requires_grad]
        if not trainable_params: print("Warning: No trainable parameters found."); return
        optimizer = AdamW(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scaler = GradScaler(enabled=(self.device == 'cuda')); best_val_iou_per_class = {cid: 0.0 for cid in classes_to_train}; overall_best_val_iou = 0.0
        optimizer.zero_grad(); global_step_counter = 0
        for class_id in classes_to_train:
            class_name = self.class_names[class_id]; print(f"\n--- Training Class: {class_name} (ID: {class_id}) ---"); mean_iou = 0.0
            for step in range(1, steps_per_class + 1):
                self.predictor.model.train()
                with autocast(device_type=self.device.split(':')[0], enabled=(self.device == 'cuda')):
                    image, mask, input_point, num_masks = self.read_batch(self.train_data, visualize=(step == 1 and class_id == classes_to_train[0]), process_specific_class=class_id, apply_augmentation=True)
                    if (image is None or mask is None or num_masks == 0 or not isinstance(input_point, np.ndarray) or input_point.size == 0): continue
                    input_label = np.ones((num_masks, 1), dtype=np.int32)
                    try: self.predictor.set_image(image)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error setting image: {e}"); continue
                    if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray): print(f"Class {class_name}, Step {step}: Invalid point/label type."); continue
                    try: mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error preparing prompts: {e}"); continue
                    if (unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0): continue
                    try: sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error generating embeddings: {e}"); continue
                    batched_mode = False
                    if not hasattr(self.predictor, '_features') or "high_res_feats" not in self.predictor._features: print(f"Class {class_name}, Step {step}: Features missing."); continue
                    high_res_features = [f[-1].unsqueeze(0) for f in self.predictor._features["high_res_feats"]]
                    if "image_embed" not in self.predictor._features: print(f"Class {class_name}, Step {step}: Image embed missing."); continue
                    try: low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder( image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0), image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features )
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error decoding mask: {e}"); continue
                    if not hasattr(self.predictor, '_orig_hw') or not self.predictor._orig_hw: print(f"Class {class_name}, Step {step}: Orig HW missing."); continue
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(self.device); prd_mask_logits = prd_masks[:, 0]
                    if gt_mask.ndim == 2: gt_mask = gt_mask.unsqueeze(0)
                    elif gt_mask.ndim != 3: continue
                    if prd_mask_logits.ndim == 2: prd_mask_logits = prd_mask_logits.unsqueeze(0)
                    elif prd_mask_logits.ndim != 3: continue
                    if gt_mask.shape[0] != prd_mask_logits.shape[0]: continue
                    prd_mask = torch.sigmoid(prd_mask_logits)
                    if gt_mask.numel() == 0 or gt_mask.shape[1] == 0 or gt_mask.shape[2] == 0: continue
                    gt_sum = gt_mask.sum(); gt_area = float(gt_mask.shape[1] * gt_mask.shape[2])
                    if gt_area == 0: continue
                    pos_ratio = gt_sum / gt_area; pos_weight = torch.clamp(1.0 - pos_ratio, min=0.1, max=0.9)
                    seg_loss = (-pos_weight * gt_mask * torch.log(prd_mask + 1e-8) - (1.0 - pos_weight) * (1.0 - gt_mask) * torch.log(1.0 - prd_mask + 1e-8)).mean()
                    with torch.no_grad(): gt_mask_bool = gt_mask > 0.5; prd_mask_bool = prd_mask > 0.5; inter = (gt_mask_bool & prd_mask_bool).sum(dim=(1, 2)).float(); union = (gt_mask_bool | prd_mask_bool).sum(dim=(1, 2)).float(); iou = inter / (union + 1e-8)
                    if prd_scores.ndim != 2 or prd_scores.shape[0] != iou.shape[0]: score_loss = torch.tensor(0.0, device=self.device)
                    else: score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    dice_score = (2.0 * inter) / (gt_mask_bool.sum(dim=(1, 2)).float() + prd_mask_bool.sum(dim=(1, 2)).float() + 1e-8); dice_loss = 1.0 - dice_score.mean()
                    if not torch.isfinite(seg_loss): seg_loss = torch.tensor(0.0, device=self.device)
                    if not torch.isfinite(score_loss): score_loss = torch.tensor(0.0, device=self.device)
                    if not torch.isfinite(dice_loss): dice_loss = torch.tensor(0.0, device=self.device)
                    loss = seg_loss * 0.7 + score_loss * 0.2 + dice_loss * 0.1; loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (step % accumulation_steps == 0) or (step == steps_per_class):
                    global_step_counter += 1; scaler.unscale_(optimizer);
                    if trainable_params: torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                with torch.no_grad(): current_iou = float(iou.mean().cpu().detach()) if torch.isfinite(iou.mean()) else 0.0
                if step == 1: mean_iou = current_iou
                else: mean_iou = (1.0 - 0.01) * mean_iou + 0.01 * current_iou
                if step % 100 == 0: current_lr = optimizer.param_groups[0]['lr']; log_loss = loss.item() * accumulation_steps; print(f"Class {class_name}, Step {step}/{steps_per_class}: Train IoU (EMA) = {mean_iou:.4f}, Batch IoU = {current_iou:.4f}, Loss = {log_loss:.4f}, LR = {current_lr:.6f}")
                if (step % validation_interval == 0 or step == steps_per_class) and self.test_data:
                    val_iou = self._validate_class(class_id, num_samples=min(50, len(self.test_data)))
                    print(f"--- Class {class_name}, Step {step}: Validation IoU = {val_iou:.4f} ---")
                    model_dir = os.path.dirname(model_name_prefix);
                    if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                    best_val_iou_per_class[class_id] = max(best_val_iou_per_class[class_id], val_iou)
                    if val_iou > overall_best_val_iou and model_dir and os.path.exists(model_dir):
                         overall_best_val_iou = val_iou; best_overall_model_path = f"{model_name_prefix}_best.pth"
                         try: torch.save(self.predictor.model.state_dict(), best_overall_model_path); print(f"Saved new overall best model to {best_overall_model_path}")
                         except Exception as e: print(f"Error saving overall best model: {e}")
                if global_step_counter > 0 and global_step_counter % checkpoint_interval == 0 and (step % accumulation_steps == 0 or step == steps_per_class):
                    model_dir = os.path.dirname(model_name_prefix);
                    if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                    if model_dir and os.path.exists(model_dir):
                        checkpoint_name = f"{model_name_prefix}_step_{global_step_counter}.pth";
                        try: torch.save(self.predictor.model.state_dict(), checkpoint_name); # print(f"Saved checkpoint: {checkpoint_name}")
                        except Exception as e: print(f"Error saving checkpoint: {e}")
        print(f"\n--- Multi-Class Training Completed ---"); print(f"Best validation IoU per class:");
        for cid in classes_to_train: print(f"  {self.class_names[cid]}: {best_val_iou_per_class[cid]:.4f}")
        print(f"Overall best validation IoU: {overall_best_val_iou:.4f}")
        model_dir = os.path.dirname(model_name_prefix);
        if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
        if model_dir and os.path.exists(model_dir):
            final_model_path = f"{model_name_prefix}_final.pth";
            try: torch.save(self.predictor.model.state_dict(), final_model_path); print(f"Final model saved: {final_model_path}")
            except Exception as e: print(f"Error saving final model: {e}")
        if overall_best_val_iou > 0: print(f"Overall best model saved at: {model_name_prefix}_best.pth")

    def _validate_class( self, class_id: int, num_samples: int = 20 ) -> float:
        """ Run validation for a specific class. (Keep implementation) """
        if not self.test_data: print(f"Warning: No test data for validating class {class_id}."); return 0.0
        if not self.predictor: print(f"Warning: Predictor not initialized."); return 0.0
        self.predictor.model.eval(); total_iou = 0.0; valid_samples_processed = 0; valid_samples_with_iou = 0
        actual_samples_to_run = min(num_samples, len(self.test_data)); indices = np.random.choice(len(self.test_data), actual_samples_to_run, replace=False)
        with torch.no_grad():
            for i in indices:
                entry = self.test_data[i];
                try: image, mask, input_point, num_masks = self.read_batch([entry], visualize=False, process_specific_class=class_id, apply_augmentation=False)
                except Exception as e: print(f"Validation Class {class_id} Error reading batch {i}: {e}"); continue
                if (image is None or mask is None or num_masks == 0 or not isinstance(input_point, np.ndarray) or input_point.size == 0): continue
                valid_samples_processed += 1; input_label = np.ones((num_masks, 1), dtype=np.int32)
                try: self.predictor.set_image(image)
                except Exception as e: print(f"Validation Class {class_id} Error setting image {i}: {e}"); continue
                try:
                     point_coords_np = input_point[:, 0, :]; point_labels_np = input_label[:, 0]
                     if not isinstance(point_coords_np, np.ndarray) or not isinstance(point_labels_np, np.ndarray): print(f"Validation Class {class_id} Error: Invalid point types {i}."); continue
                     pred_masks_val, scores_val, logits_val = self.predictor.predict(point_coords=point_coords_np, point_labels=point_labels_np, multimask_output=False)
                except Exception as e: print(f"Validation Class {class_id} Error prediction {i}: {e}"); continue
                if pred_masks_val is None or len(pred_masks_val) == 0: continue
                pred_mask_val = pred_masks_val[0].astype(bool); gt_mask_val = mask[0].astype(bool)
                intersection = np.logical_and(gt_mask_val, pred_mask_val).sum(); union = np.logical_or(gt_mask_val, pred_mask_val).sum()
                if union > 0: iou = intersection / union; total_iou += iou; valid_samples_with_iou += 1
        self.predictor.model.train(); avg_iou = total_iou / max(1, valid_samples_with_iou); return avg_iou


    # --- NEW INFERENCE METHOD HELPER: _read_inference_image ---
    def _read_inference_image(self, image_path: Optional[str], json_path: str, target_size: int = 1024) -> Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """ Reads image from file or JSON, resizes, returns RGB image and shapes."""
        image = None; abs_json_path = os.path.abspath(json_path); abs_image_path = os.path.abspath(image_path) if image_path else None
        try:
            json_contains_image = False
            if os.path.isfile(abs_json_path):
                 with open(abs_json_path, 'r') as f: image_data_base64 = json.load(f).get("imageData")
                 if image_data_base64: json_contains_image = True; image_data = base64.b64decode(image_data_base64); image_np = np.frombuffer(image_data, np.uint8); image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                 if image is not None: print(f"Loaded image from JSON: {abs_json_path}")
            if image is None and abs_image_path and os.path.isfile(abs_image_path): image = cv2.imread(abs_image_path)
            if image is None: raise ValueError(f"Could not load image for {abs_json_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image_rgb.shape[:2]; h, w = original_shape; scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"Resized image to: ({new_h}, {new_w})")
            return resized_image_rgb, original_shape, (new_h, new_w)
        except Exception as e: print(f"Error loading/resizing image: {e}"); return None

    # --- NEW INFERENCE METHOD HELPER: _sample_points ---
    def _sample_points(self, mask: np.ndarray, num_points: int) -> Optional[np.ndarray]:
        """ Samples N points from the foreground of a binary mask. """
        if mask is None or mask.sum() == 0: print("Warning: Cannot sample points from empty mask."); return None
        coords = np.argwhere(mask > 0) # y, x
        if len(coords) == 0: print("Warning: No foreground pixels found after thresholding mask."); return None
        replace = len(coords) < num_points
        if replace: print(f"Warning: Requested {num_points} points, but only {len(coords)} available. Sampling with replacement.")
        chosen_indices = np.random.choice(len(coords), num_points, replace=replace)
        points_yx = coords[chosen_indices]; points_xy = points_yx[:, ::-1] # Convert to x, y format
        return points_xy.astype(np.int32)

    # --- NEW INFERENCE METHOD (Using existing predictor) ---
    def inference(
        self,
        image_path: Optional[str],     # Path to image file (can be None if in JSON)
        json_path: str,                # Path to annotation JSON file (for GT mask and maybe image)
        checkpoint_path: str,          # Path to the FINE-TUNED .pth file
        num_points: int = 10,          # Number of points to sample from GT mask as prompt
        target_size: int = 1024,
        device: Optional[str] = None   # Optional: override device
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Performs inference using a specified fine-tuned checkpoint.
        Assumes self.predictor is ALREADY INITIALIZED via initialize_model().
        Loads fine-tuned weights into the existing model, preprocesses data,
        samples points from GT, predicts.

        Args:
            image_path: Path to the image file. Can be None if image data is in JSON.
            json_path: Path to the annotation JSON file. Used for GT mask and potentially image data.
            checkpoint_path: Path to the fine-tuned model weights (.pth file).
            num_points: Number of points to sample from GT mask as prompt.
            target_size: Target size for resizing images.
            device: Device to run on ('cuda' or 'cpu'). Uses self.device if None.

        Returns:
            Tuple containing:
                - Resized RGB image (np.ndarray) or None on failure.
                - Resized Ground Truth binary mask (np.ndarray) or None on failure.
                - Predicted binary mask (np.ndarray) or None on failure.
        """
        print("-" * 30); print("Starting Inference using fine-tuned model...")

        # 1. Check if predictor exists
        if self.predictor is None or self.sam_model is None:
            print("Error: Predictor not initialized. Call initialize_model() before inference.")
            return None, None, None

        _device = device if device else self.device
        if _device == "cuda" and not torch.cuda.is_available(): print("Warning: CUDA requested but not available, using CPU."); _device = "cpu"
        print(f"Using device: {_device}")
        abs_checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Loading fine-tuned weights: {abs_checkpoint_path}")

        # 2. Load Fine-tuned Weights into EXISTING model
        try:
            if not os.path.isfile(abs_checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {abs_checkpoint_path}")
            state_dict = torch.load(abs_checkpoint_path, map_location=_device)
            # Load state dict with strict=False initially if architecture might slightly differ (e.g., unused pretrained params)
            # Or ensure the saved checkpoint matches the architecture exactly
            missing_keys, unexpected_keys = self.sam_model.load_state_dict(state_dict, strict=True) # Use strict=True if sure
            if unexpected_keys: print(f"Warning: Unexpected keys found in checkpoint: {unexpected_keys}")
            if missing_keys: print(f"Warning: Missing keys in model state_dict: {missing_keys}")
            print("Fine-tuned weights loaded successfully.")
            self.sam_model.eval() # Set model to eval mode
        except Exception as e: print(f"Error loading fine-tuned state dict: {e}"); return None, None, None

        # 3. Load and Preprocess Image and GT Mask
        read_result = self._read_inference_image(image_path, json_path, target_size)
        if read_result is None: return None, None, None
        resized_image_rgb, original_shape, (new_h, new_w) = read_result

        gt_binary_mask = None
        try:
            abs_json_path = os.path.abspath(json_path)
            gt_multi_class_mask = self._parse_json_annotation(abs_json_path, original_shape) # Use self method
            if gt_multi_class_mask is None: raise ValueError("Failed to parse GT annotation.")
            gt_multi_class_mask_resized = cv2.resize(gt_multi_class_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            gt_binary_mask = (gt_multi_class_mask_resized > 0).astype(np.uint8) # Use binary GT for sampling/comparison
        except Exception as e: print(f"Error loading/processing GT mask: {e}"); return resized_image_rgb, None, None

        # 4. Sample Points from GT Mask
        input_points = self._sample_points(gt_binary_mask, num_points) # Use self method
        if input_points is None: print("Could not sample points. Cannot predict."); return resized_image_rgb, gt_binary_mask, None
        print(f"Sampled {input_points.shape[0]} points from GT mask.")
        input_labels = np.ones(input_points.shape[0], dtype=np.int32) # Positive labels

        # 5. Perform Prediction using the existing predictor
        pred_binary_mask = None
        try:
            print("Setting image..."); self.predictor.set_image(resized_image_rgb)
            print(f"Running prediction...");
            with torch.no_grad():
                 pred_masks, pred_scores, pred_logits = self.predictor.predict(
                    point_coords=input_points, # Shape [N, 2]
                    point_labels=input_labels,  # Shape [N]
                    multimask_output=False # Get single best mask per point
                )
            print(f"Prediction complete.")
            if pred_masks is None or len(pred_masks) == 0: print("Warning: Model returned no masks."); pred_binary_mask = np.zeros_like(gt_binary_mask)
            else:
                # Combine masks using a standard loop for clarity
                combined_pred_mask = np.zeros_like(pred_masks[0], dtype=bool)
                for p_mask in pred_masks: # Iterate through predicted masks
                    combined_pred_mask |= p_mask.astype(bool) # Use '|=' for logical OR assignment
                pred_binary_mask = combined_pred_mask.astype(np.uint8)
                print("Processed predicted masks.")

        except Exception as e: print(f"Error during prediction: {e}"); import traceback; traceback.print_exc(); return resized_image_rgb, gt_binary_mask, None

        print("Inference method finished.")
        # Return image, GT mask (binary), Prediction mask (binary)
        return resized_image_rgb, gt_binary_mask, pred_binary_mask

    # --- Keep train_multiclass, _validate_class methods ---
    def train_multiclass( self, steps_per_class: int = 500, learning_rate: float = 1e-5, weight_decay: float = 2e-4, accumulation_steps: int = 8, scheduler_step_size: int = 100, scheduler_gamma: float = 0.5, checkpoint_interval: int = 200, validation_interval: int = 200, model_name_prefix: str = "./models/heart_chambers_sam_multiclass" ) -> None:
        """ Train the SAM model using a class-by-class approach. (Keep implementation) """
        if not self.predictor: raise ValueError("Model not initialized."); print("Call initialize_model() first.")
        if not self.train_data: raise ValueError("Training data not prepared."); print("Call prepare_data() first.")
        classes_to_train = [cid for cid in self.chamber_classes.values() if cid > 0]
        if not classes_to_train: print("Warning: No classes found to train."); return
        print(f"\n--- Starting Training (Multi-Class Sequential Mode) ---"); print(f"Classes: {[self.class_names[i] for i in classes_to_train]}"); print(f"Steps per class: {steps_per_class}")
        trainable_params = [p for p in self.predictor.model.parameters() if p.requires_grad]
        if not trainable_params: print("Warning: No trainable parameters found."); return
        optimizer = AdamW(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scaler = GradScaler(enabled=(self.device == 'cuda')); best_val_iou_per_class = {cid: 0.0 for cid in classes_to_train}; overall_best_val_iou = 0.0
        optimizer.zero_grad(); global_step_counter = 0
        for class_id in classes_to_train:
            class_name = self.class_names[class_id]; print(f"\n--- Training Class: {class_name} (ID: {class_id}) ---"); mean_iou = 0.0
            for step in range(1, steps_per_class + 1):
                self.predictor.model.train()
                with autocast(device_type=self.device.split(':')[0], enabled=(self.device == 'cuda')):
                    image, mask, input_point, num_masks = self.read_batch(self.train_data, visualize=(step == 1 and class_id == classes_to_train[0]), process_specific_class=class_id, apply_augmentation=True)
                    if (image is None or mask is None or num_masks == 0 or not isinstance(input_point, np.ndarray) or input_point.size == 0): continue
                    input_label = np.ones((num_masks, 1), dtype=np.int32)
                    try: self.predictor.set_image(image)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error setting image: {e}"); continue
                    if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray): print(f"Class {class_name}, Step {step}: Invalid point/label type."); continue
                    try: mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error preparing prompts: {e}"); continue
                    if (unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0): continue
                    try: sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error generating embeddings: {e}"); continue
                    batched_mode = False
                    if not hasattr(self.predictor, '_features') or "high_res_feats" not in self.predictor._features: print(f"Class {class_name}, Step {step}: Features missing."); continue
                    high_res_features = [f[-1].unsqueeze(0) for f in self.predictor._features["high_res_feats"]]
                    if "image_embed" not in self.predictor._features: print(f"Class {class_name}, Step {step}: Image embed missing."); continue
                    try: low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder( image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0), image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features )
                    except Exception as e: print(f"Class {class_name}, Step {step}: Error decoding mask: {e}"); continue
                    if not hasattr(self.predictor, '_orig_hw') or not self.predictor._orig_hw: print(f"Class {class_name}, Step {step}: Orig HW missing."); continue
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(self.device); prd_mask_logits = prd_masks[:, 0]
                    if gt_mask.ndim == 2: gt_mask = gt_mask.unsqueeze(0)
                    elif gt_mask.ndim != 3: continue
                    if prd_mask_logits.ndim == 2: prd_mask_logits = prd_mask_logits.unsqueeze(0)
                    elif prd_mask_logits.ndim != 3: continue
                    if gt_mask.shape[0] != prd_mask_logits.shape[0]: continue
                    prd_mask = torch.sigmoid(prd_mask_logits)
                    if gt_mask.numel() == 0 or gt_mask.shape[1] == 0 or gt_mask.shape[2] == 0: continue
                    gt_sum = gt_mask.sum(); gt_area = float(gt_mask.shape[1] * gt_mask.shape[2])
                    if gt_area == 0: continue
                    pos_ratio = gt_sum / gt_area; pos_weight = torch.clamp(1.0 - pos_ratio, min=0.1, max=0.9)
                    seg_loss = (-pos_weight * gt_mask * torch.log(prd_mask + 1e-8) - (1.0 - pos_weight) * (1.0 - gt_mask) * torch.log(1.0 - prd_mask + 1e-8)).mean()
                    with torch.no_grad(): gt_mask_bool = gt_mask > 0.5; prd_mask_bool = prd_mask > 0.5; inter = (gt_mask_bool & prd_mask_bool).sum(dim=(1, 2)).float(); union = (gt_mask_bool | prd_mask_bool).sum(dim=(1, 2)).float(); iou = inter / (union + 1e-8)
                    if prd_scores.ndim != 2 or prd_scores.shape[0] != iou.shape[0]: score_loss = torch.tensor(0.0, device=self.device)
                    else: score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    dice_score = (2.0 * inter) / (gt_mask_bool.sum(dim=(1, 2)).float() + prd_mask_bool.sum(dim=(1, 2)).float() + 1e-8); dice_loss = 1.0 - dice_score.mean()
                    if not torch.isfinite(seg_loss): seg_loss = torch.tensor(0.0, device=self.device)
                    if not torch.isfinite(score_loss): score_loss = torch.tensor(0.0, device=self.device)
                    if not torch.isfinite(dice_loss): dice_loss = torch.tensor(0.0, device=self.device)
                    loss = seg_loss * 0.7 + score_loss * 0.2 + dice_loss * 0.1; loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (step % accumulation_steps == 0) or (step == steps_per_class):
                    global_step_counter += 1; scaler.unscale_(optimizer);
                    if trainable_params: torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                with torch.no_grad(): current_iou = float(iou.mean().cpu().detach()) if torch.isfinite(iou.mean()) else 0.0
                if step == 1: mean_iou = current_iou
                else: mean_iou = (1.0 - 0.01) * mean_iou + 0.01 * current_iou
                if step % 100 == 0: current_lr = optimizer.param_groups[0]['lr']; log_loss = loss.item() * accumulation_steps; print(f"Class {class_name}, Step {step}/{steps_per_class}: Train IoU (EMA) = {mean_iou:.4f}, Batch IoU = {current_iou:.4f}, Loss = {log_loss:.4f}, LR = {current_lr:.6f}")
                if (step % validation_interval == 0 or step == steps_per_class) and self.test_data:
                    val_iou = self._validate_class(class_id, num_samples=min(50, len(self.test_data)))
                    print(f"--- Class {class_name}, Step {step}: Validation IoU = {val_iou:.4f} ---")
                    model_dir = os.path.dirname(model_name_prefix);
                    if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                    best_val_iou_per_class[class_id] = max(best_val_iou_per_class[class_id], val_iou)
                    if val_iou > overall_best_val_iou and model_dir and os.path.exists(model_dir):
                         overall_best_val_iou = val_iou; best_overall_model_path = f"{model_name_prefix}_best.pth"
                         try: torch.save(self.predictor.model.state_dict(), best_overall_model_path); print(f"Saved new overall best model to {best_overall_model_path}")
                         except Exception as e: print(f"Error saving overall best model: {e}")
                if global_step_counter > 0 and global_step_counter % checkpoint_interval == 0 and (step % accumulation_steps == 0 or step == steps_per_class):
                    model_dir = os.path.dirname(model_name_prefix);
                    if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
                    if model_dir and os.path.exists(model_dir):
                        checkpoint_name = f"{model_name_prefix}_step_{global_step_counter}.pth";
                        try: torch.save(self.predictor.model.state_dict(), checkpoint_name); # print(f"Saved checkpoint: {checkpoint_name}")
                        except Exception as e: print(f"Error saving checkpoint: {e}")
        print(f"\n--- Multi-Class Training Completed ---"); print(f"Best validation IoU per class:");
        for cid in classes_to_train: print(f"  {self.class_names[cid]}: {best_val_iou_per_class[cid]:.4f}")
        print(f"Overall best validation IoU: {overall_best_val_iou:.4f}")
        model_dir = os.path.dirname(model_name_prefix);
        if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
        if model_dir and os.path.exists(model_dir):
            final_model_path = f"{model_name_prefix}_final.pth";
            try: torch.save(self.predictor.model.state_dict(), final_model_path); print(f"Final model saved: {final_model_path}")
            except Exception as e: print(f"Error saving final model: {e}")
        if overall_best_val_iou > 0: print(f"Overall best model saved at: {model_name_prefix}_best.pth")

    def _validate_class( self, class_id: int, num_samples: int = 20 ) -> float:
        """ Run validation for a specific class. (Keep implementation) """
        if not self.test_data: print(f"Warning: No test data for validating class {class_id}."); return 0.0
        if not self.predictor: print(f"Warning: Predictor not initialized."); return 0.0
        self.predictor.model.eval(); total_iou = 0.0; valid_samples_processed = 0; valid_samples_with_iou = 0
        actual_samples_to_run = min(num_samples, len(self.test_data)); indices = np.random.choice(len(self.test_data), actual_samples_to_run, replace=False)
        with torch.no_grad():
            for i in indices:
                entry = self.test_data[i];
                try: image, mask, input_point, num_masks = self.read_batch([entry], visualize=False, process_specific_class=class_id, apply_augmentation=False)
                except Exception as e: print(f"Validation Class {class_id} Error reading batch {i}: {e}"); continue
                if (image is None or mask is None or num_masks == 0 or not isinstance(input_point, np.ndarray) or input_point.size == 0): continue
                valid_samples_processed += 1; input_label = np.ones((num_masks, 1), dtype=np.int32)
                try: self.predictor.set_image(image)
                except Exception as e: print(f"Validation Class {class_id} Error setting image {i}: {e}"); continue
                try:
                     point_coords_np = input_point[:, 0, :]; point_labels_np = input_label[:, 0]
                     if not isinstance(point_coords_np, np.ndarray) or not isinstance(point_labels_np, np.ndarray): print(f"Validation Class {class_id} Error: Invalid point types {i}."); continue
                     pred_masks_val, scores_val, logits_val = self.predictor.predict(point_coords=point_coords_np, point_labels=point_labels_np, multimask_output=False)
                except Exception as e: print(f"Validation Class {class_id} Error prediction {i}: {e}"); continue
                if pred_masks_val is None or len(pred_masks_val) == 0: continue
                pred_mask_val = pred_masks_val[0].astype(bool); gt_mask_val = mask[0].astype(bool)
                intersection = np.logical_and(gt_mask_val, pred_mask_val).sum(); union = np.logical_or(gt_mask_val, pred_mask_val).sum()
                if union > 0: iou = intersection / union; total_iou += iou; valid_samples_with_iou += 1
        self.predictor.model.train(); avg_iou = total_iou / max(1, valid_samples_with_iou); return avg_iou

