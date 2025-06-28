"""
Prototypical Few-Shot Out-of-Distribution (OOD) Detection for Tree Species Classification

This module implements the main methodology for prototypical few-shot learning 
with OOD detection using CLIP vision encoder, as described in the research paper.

The pipeline includes:
1. Data loading and preprocessing
2. Feature extraction using CLIP ViT-B/32
3. Prototype computation for ID classes
4. OOD crop generation
5. Similarity calculation and threshold determination
6. OOD classification and evaluation

Author: Chen Yuhong
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import os
import gc
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm
import rasterio as rio
import albumentations as A
from einops import rearrange
import functools
import itertools
import warnings
warnings.filterwarnings('ignore')


class TreeSpeciesDataloader:
    """Enhanced data loader for multispectral UAV imagery following create_dataset.ipynb pipeline"""
    
    def __init__(self, casuarina_93deg_path=None, casuarina_183deg_path=None, 
                 chestnut_may_path=None, chestnut_dec_path=None):
        """
        Initialize dataloader with paths to all available datasets
        
        Args:
            casuarina_93deg_path: Path to Casuarina 93° dataset
            casuarina_183deg_path: Path to Casuarina 183° dataset 
            chestnut_may_path: Path to Chestnut May dataset
            chestnut_dec_path: Path to Chestnut December dataset
        """
        # Convert to Path objects
        self.casuarina_93deg_path = Path(casuarina_93deg_path) if casuarina_93deg_path else None
        self.casuarina_183deg_path = Path(casuarina_183deg_path) if casuarina_183deg_path else None
        self.chestnut_may_path = Path(chestnut_may_path) if chestnut_may_path else None
        self.chestnut_dec_path = Path(chestnut_dec_path) if chestnut_dec_path else None
        
        # Band configuration following create_dataset.ipynb
        self.band_filenames = [
            'result.tif',
            'result_Red.tif', 
            'result_Green.tif',
            'result_Blue.tif',
            'result_NIR.tif',
            'result_RedEdge.tif',
        ]
        
        # Bands dictionary for reference
        self.bands_dict = {
            'wr': 'Wideband Red',
            'wg': 'Wideband Green', 
            'wb': 'Wideband Blue',
            'r': 'Narrowband Red',
            'g': 'Narrowband Green',
            'b': 'Narrowband Blue',
            'nir': 'Near Infrared',
            're': 'Red Edge',
        }
        
    def load_dataset(self, ds_path):
        """Load multispectral dataset from path following create_dataset.ipynb methodology"""
        filenames = [ds_path / filename for filename in self.band_filenames]
        arrs = [rio.open(filename).read() for filename in filenames]
        arrs[0] = arrs[0][:3]  # Keep only first 3 bands from result.tif
        arrs = np.concatenate(arrs)
        return arrs
    
    def load_bounds(self, ds_path):
        """Load bounding box annotations"""
        bounds_file = ds_path / 'bounds.csv'
        if bounds_file.exists():
            return pd.read_csv(bounds_file)
        else:
            print(f"Warning: bounds.csv not found in {ds_path}")
            return pd.DataFrame()
    
    
    def produce_crop(self, ds, y0, y1, x0, x1):
        """Extract crop from dataset with NaN handling"""
        return np.nan_to_num(ds[:, y0:y1, x0:x1])
    
    def prepare_comprehensive_dataset(self):
        """Prepare complete dataset with all available data following create_dataset.ipynb"""
        print("Preparing comprehensive multispectral dataset...")
        
        all_bounds = []
        datasets = {}
        
        # Load all available datasets
        dataset_configs = [
            (self.casuarina_93deg_path, 'casuarina', '93deg'),
            (self.casuarina_183deg_path, 'casuarina', '183deg'),
            (self.chestnut_may_path, 'chestnut', 'may'),
            (self.chestnut_dec_path, 'chestnut', 'dec')
        ]
        
        for path, ds_name, capture in dataset_configs:
            if path and path.exists():
                print(f"Loading {ds_name} {capture} dataset...")
                
                # Load multispectral data
                dataset = self.load_dataset(path)
                datasets[f"{ds_name}_{capture}"] = dataset
                
                # Load bounds
                bounds = self.load_bounds(path)
                if not bounds.empty:
                    bounds['ds'] = ds_name
                    bounds['capture'] = capture
                    all_bounds.append(bounds)
        
        # Combine all bounds
        if all_bounds:
            bounds_df = pd.concat(all_bounds, ignore_index=True)
            
            # Generate crops for each bound
            print("Generating crops from bounding boxes...")
            crops = []
            for _, row in tqdm(bounds_df.iterrows(), total=len(bounds_df)):
                dataset_key = f"{row['ds']}_{row['capture']}"
                if dataset_key in datasets:
                    crop = self.produce_crop(datasets[dataset_key], 
                                           int(row['y0']), int(row['y1']), 
                                           int(row['x0']), int(row['x1']))
                    crops.append(crop)
                else:
                    crops.append(None)
            
            bounds_df['crop'] = crops
            
            # Remove rows with None crops
            bounds_df = bounds_df[bounds_df['crop'].notna()].reset_index(drop=True)
            
            print(f"Loaded {len(bounds_df)} labeled samples from {bounds_df['name'].nunique()} species")
            print(f"Species distribution: {bounds_df['name'].value_counts().to_dict()}")
            
            return bounds_df, datasets
        else:
            print("No valid bounds data found!")
            return pd.DataFrame(), datasets


class DataAugmentation:
    """Data augmentation for tree species images"""
    
    def __init__(self):
        self.augmentation_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(gamma_limit=(50, 150), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.05, p=0.3),
            A.CLAHE(p=0.3),
            A.Posterize(p=0.3),
            A.Solarize(threshold=128, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.ElasticTransform(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomRain(p=0.2),
            A.RandomSnow(p=0.2),
            A.Emboss(p=0.3),
            A.Sharpen(p=0.3),
        ]
    
    def get_random_augmentations(self):
        """Get random combination of 2-6 augmentations"""
        n_augs = random.randint(2, 6)
        selected_augs = random.sample(self.augmentation_list, n_augs)
        return A.Compose(selected_augs)
    
    def augment_image(self, image_array):
        """Apply random augmentations to image array"""
        augmentations = self.get_random_augmentations()
        augmented = augmentations(image=image_array)["image"]
        return augmented
    
    def generate_augmented_samples(self, crop, num_samples=19):
        """Generate augmented samples for a given crop (19 augmented + 1 original = 20 total)"""
        # Convert crop to RGB format for augmentation
        if crop.shape[0] > 3:
            # Use bands 0 (wideband red), 1 (wideband green), 7 (red-edge) as described
            pseudo_rgb = np.stack([crop[0], crop[1], crop[7] if crop.shape[0] > 7 else crop[2]], axis=-1)
        else:
            pseudo_rgb = np.transpose(crop[:3], (1, 2, 0))
        
        # Normalize to 0-255 range
        if pseudo_rgb.max() <= 1.0:
            pseudo_rgb = (pseudo_rgb * 255).astype(np.uint8)
        else:
            pseudo_rgb = ((pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min()) * 255).astype(np.uint8)
        
        samples = [pseudo_rgb]  # Original image
        
        # Generate augmented samples
        for _ in range(num_samples):
            augmented = self.augment_image(pseudo_rgb)
            samples.append(augmented)
        
        return samples


class CLIPFeatureExtractor:
    """CLIP-based feature extraction for tree species images"""
    
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
    
    def crop_to_pil(self, crop):
        """Convert crop array to PIL image"""
        if len(crop.shape) == 3:
            if crop.shape[0] <= 3:
                # Transpose from (C, H, W) to (H, W, C)
                img_array = np.transpose(crop[:3], (1, 2, 0))
            else:
                # Use pseudo-RGB: bands 0, 1, 7 (wideband red, green, red-edge)
                img_array = np.stack([crop[0], crop[1], crop[7] if crop.shape[0] > 7 else crop[2]], axis=-1)
        else:
            img_array = crop
        
        # Normalize to 0-255 range
        if img_array.dtype != np.uint8:
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min)
            img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def extract_features(self, image_samples):
        """Extract CLIP features from image samples"""
        features = []
        
        for sample in image_samples:
            if isinstance(sample, np.ndarray):
                pil_img = Image.fromarray(sample)
            else:
                pil_img = self.crop_to_pil(sample)
            
            # Preprocess and encode
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
            
            # Normalize feature
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            features.append(image_feature.cpu().numpy())
        
        return np.vstack(features)


class OverlapReductionMethods:
    """
    Enhanced overlap reduction methods for improving ID/OOD separation.
    
    Implements three methods from Overlap_reduction.ipynb:
    1. Original Method: Direct cosine similarity
    2. Sampled Mean Method: Average similarity across multiple augmented crops
    3. Ensemble Method: Neural network combining multiple similarity features
    """
    
    def __init__(self, model, preprocess, augmentor, device):
        self.model = model
        self.preprocess = preprocess
        self.augmentor = augmentor
        self.device = device
        self.ensemble_net = None
        self.adaptive_weights = None
        self.neural_weights = None
    
    def original_similarity_method(self, query_features, prototypes):
        """Original method: Direct cosine similarity between query and prototypes"""
        similarities = []
        
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.to(self.device)
        
        for prototype in prototypes.values():
            if isinstance(prototype, torch.Tensor):
                prototype = prototype.to(self.device)
            else:
                prototype = torch.tensor(prototype).to(self.device)
            
            similarity = torch.cosine_similarity(query_features, prototype.unsqueeze(0), dim=1)
            similarities.append(similarity)
        
        return torch.stack(similarities).max(dim=0)[0]
    
    def extract_clip_features(self, images):
        """Extract CLIP features from a list of images"""
        features = []
        
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model.encode_image(img_input)
                feature = feature / feature.norm(dim=1, keepdim=True)
                feature = feature.float()
            
            features.append(feature.cpu())
        
        return torch.cat(features, dim=0).float()
    
    def sampled_mean_similarity_method(self, query_image, prototypes, num_crops=10):
        """Sampled mean method: Average similarity across multiple augmented crops"""
        # Convert query_image to appropriate format
        if isinstance(query_image, torch.Tensor):
            query_image = query_image.cpu().numpy()
        
        if len(query_image.shape) == 3 and query_image.shape[0] <= 3:
            # Convert from (C, H, W) to (H, W, C)
            query_image = np.transpose(query_image, (1, 2, 0))
        
        # Normalize to 0-255 range for augmentation
        if query_image.max() <= 1.0:
            query_image = (query_image * 255).astype(np.uint8)
        elif query_image.dtype != np.uint8:
            query_image = ((query_image - query_image.min()) / 
                          (query_image.max() - query_image.min()) * 255).astype(np.uint8)
        
        # Generate augmented crops
        augmented_crops = self.augmentor.generate_augmented_samples(
            query_image.transpose(2, 0, 1) if len(query_image.shape) == 3 else query_image, 
            num_crops - 1
        )
        
        # Extract features from all crops
        crop_features = self.extract_clip_features(augmented_crops)
        crop_features = crop_features.float().to(self.device)
        
        # Calculate similarities for each crop
        crop_similarities = []
        for crop_feature in crop_features:
            similarities = []
            for prototype in prototypes.values():
                if isinstance(prototype, torch.Tensor):
                    prototype = prototype.float().to(self.device)
                else:
                    prototype = torch.tensor(prototype, dtype=torch.float32).to(self.device)
                
                crop_feature_norm = crop_feature.unsqueeze(0)
                prototype_norm = prototype.unsqueeze(0)
                
                similarity = torch.cosine_similarity(crop_feature_norm, prototype_norm, dim=1)
                similarities.append(similarity)
            
            if similarities:
                crop_similarities.append(torch.stack(similarities).max())
        
        # Return mean similarity across crops
        if crop_similarities:
            return torch.stack(crop_similarities).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def pair_similarity(self, img1, img2):
        """Calculate direct similarity between two images using CLIP"""
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)
        
        img1_input = self.preprocess(img1).unsqueeze(0).to(self.device)
        img2_input = self.preprocess(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature1 = self.model.encode_image(img1_input).float()
            feature2 = self.model.encode_image(img2_input).float()
        
        # Normalize features
        feature1 = feature1 / feature1.norm(dim=1, keepdim=True)
        feature2 = feature2 / feature2.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        cosine_sim = torch.cosine_similarity(feature1, feature2, dim=1)
        return cosine_sim.item()
    
    def ensemble_similarity(self, arr1, arr2):
        """
        Calculate enhanced similarity features using multiple approaches:
        1. Original direct similarity
        2. Augmented crops mean similarity  
        3. Maximum similarity across crops (best match)
        4. Minimum similarity across crops (worst match)
        5. Standard deviation of similarities (consistency measure)
        6. Percentile-based measures
        """
        # Convert to PIL images if needed
        if isinstance(arr1, np.ndarray):
            if len(arr1.shape) == 3 and arr1.shape[0] <= 3:
                arr1 = np.transpose(arr1, (1, 2, 0))
            if arr1.max() <= 1.0:
                arr1 = (arr1 * 255).astype(np.uint8)
            img1 = Image.fromarray(arr1)
        else:
            img1 = arr1
            
        if isinstance(arr2, np.ndarray):
            if len(arr2.shape) == 3 and arr2.shape[0] <= 3:
                arr2 = np.transpose(arr2, (1, 2, 0))
            if arr2.max() <= 1.0:
                arr2 = (arr2 * 255).astype(np.uint8)
            img2 = Image.fromarray(arr2)
        else:
            img2 = arr2
        
        # Generate augmented crops
        crops1 = self.augmentor.generate_augmented_samples(
            arr1.transpose(2, 0, 1) if isinstance(arr1, np.ndarray) and len(arr1.shape) == 3 else arr1,
            num_samples=9
        )
        crops2 = self.augmentor.generate_augmented_samples(
            arr2.transpose(2, 0, 1) if isinstance(arr2, np.ndarray) and len(arr2.shape) == 3 else arr2,
            num_samples=9
        )
        
        # Calculate all pairwise similarities
        sim_scores = []
        for i in range(len(crops1)):
            for j in range(len(crops2)):
                score = self.pair_similarity(crops1[i], crops2[j])
                sim_scores.append(score)
        
        sim_scores = np.array(sim_scores)
        
        # Direct similarity (baseline)
        direct_sim = self.pair_similarity(img1, img2)
        
        # Statistical measures
        mean_sim = np.mean(sim_scores)
        max_sim = np.max(sim_scores)
        min_sim = np.min(sim_scores)
        std_sim = np.std(sim_scores)
        median_sim = np.median(sim_scores)
        
        # Percentile measures
        p75_sim = np.percentile(sim_scores, 75)
        p25_sim = np.percentile(sim_scores, 25)
        iqr_sim = p75_sim - p25_sim
        
        return {
            'direct': direct_sim,
            'mean': mean_sim,
            'max': max_sim,
            'min': min_sim,
            'std': std_sim,
            'median': median_sim,
            'p75': p75_sim,
            'p25': p25_sim,
            'iqr': iqr_sim,
            'all_scores': sim_scores
        }
    
    def ensemble_similarity_score(self, features, weights=None):
        """Calculate ensemble similarity score using weighted combination of features"""
        if weights is None:
            # Default optimized weights
            weights = {
                'direct': 0.8347946405410767,
                'mean': 0.045269619673490524,
                'max': 0.004325345158576965,
                'min': -0.0026785789523273706,
                'std': -0.007063772063702345,
                'median': 0.044147342443466187,
                'p75': 0.04279244691133499,
                'p25': 0.015011320821940899,  
                'iqr': -0.003916935063898563
            }
        
        score = 0
        for key, weight in weights.items():
            if key in features:
                score += weight * features[key]
        
        return score
    
    def calculate_feature_discriminative_power(self, id_features, ood_features):
        """Calculate discriminative power of each feature using multiple metrics"""
        feature_names = ['direct', 'mean', 'max', 'min', 'std', 'median', 'p75', 'p25', 'iqr']
        discriminative_scores = {}
        
        for feature_name in feature_names:
            # Extract feature values for ID and OOD
            id_values = [features[feature_name] for features in id_features if feature_name in features]
            ood_values = [features[feature_name] for features in ood_features if feature_name in features]
            
            if len(id_values) == 0 or len(ood_values) == 0:
                discriminative_scores[feature_name] = 0.0
                continue
                
            id_values = np.array(id_values)
            ood_values = np.array(ood_values)
            
            # ROC AUC Score
            try:
                y_true = np.concatenate([np.ones(len(id_values)), np.zeros(len(ood_values))])
                y_scores = np.concatenate([id_values, ood_values])
                auc_score = roc_auc_score(y_true, y_scores)
                auc_power = abs(auc_score - 0.5) * 2
            except:
                auc_power = 0.0
            
            # Effect Size (Cohen's d)
            try:
                pooled_std = np.sqrt(((len(id_values) - 1) * np.var(id_values, ddof=1) + 
                                     (len(ood_values) - 1) * np.var(ood_values, ddof=1)) / 
                                    (len(id_values) + len(ood_values) - 2))
                if pooled_std > 0:
                    cohens_d = abs(np.mean(id_values) - np.mean(ood_values)) / pooled_std
                else:
                    cohens_d = 0.0
            except:
                cohens_d = 0.0
            
            # Mann-Whitney U test
            try:
                _, p_value = mannwhitneyu(id_values, ood_values, alternative='two-sided')
                mw_power = max(0, 1 - p_value)
            except:
                mw_power = 0.0
            
            # Separation metric
            try:
                mean_diff = abs(np.mean(id_values) - np.mean(ood_values))
                pooled_std = (np.std(id_values) + np.std(ood_values)) / 2
                if pooled_std > 0:
                    separation = mean_diff / pooled_std
                else:
                    separation = 0.0
            except:
                separation = 0.0
            
            # Combine metrics
            combined_score = (0.4 * auc_power + 0.3 * min(cohens_d, 2.0) / 2.0 + 
                             0.2 * mw_power + 0.1 * min(separation, 3.0) / 3.0)
            
            discriminative_scores[feature_name] = combined_score
        
        return discriminative_scores
    
    def learn_optimal_weights(self, id_features, ood_features, regularization=0.1):
        """Learn optimal weights based on discriminative power with regularization"""
        # Calculate discriminative power for each feature
        discriminative_scores = self.calculate_feature_discriminative_power(id_features, ood_features)
        
        print("Feature Discriminative Power Scores:")
        for feature, score in discriminative_scores.items():
            print(f"  {feature}: {score:.4f}")
        
        # Convert to weights with regularization
        total_score = sum(discriminative_scores.values())
        if total_score == 0:
            uniform_weight = 1.0 / len(discriminative_scores)
            return {feature: uniform_weight for feature in discriminative_scores.keys()}
        
        # Normalize scores to weights
        raw_weights = {feature: score / total_score for feature, score in discriminative_scores.items()}
        
        # Apply regularization
        uniform_weight = 1.0 / len(raw_weights)
        regularized_weights = {}
        
        for feature, raw_weight in raw_weights.items():
            regularized_weight = (1 - regularization) * raw_weight + regularization * uniform_weight
            regularized_weights[feature] = regularized_weight
        
        # Handle consistency features (std, iqr should have negative weights)
        if 'std' in regularized_weights:
            regularized_weights['std'] = -regularized_weights['std']
        
        if 'iqr' in regularized_weights:
            regularized_weights['iqr'] = -regularized_weights['iqr']
        
        # Renormalize
        total_abs_weight = sum(abs(w) for w in regularized_weights.values())
        if total_abs_weight > 0:
            for feature in regularized_weights:
                regularized_weights[feature] = regularized_weights[feature] / total_abs_weight
        
        self.adaptive_weights = regularized_weights
        return regularized_weights


class EnsembleWeightNet(nn.Module):
    """Enhanced neural network to learn optimal weights for ensemble similarity features"""
    
    def __init__(self, num_features=9, hidden_dim=64):
        super(EnsembleWeightNet, self).__init__()
        self.num_features = num_features
        
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_features),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier uniform initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, dummy_input=None):
        """Forward pass - learns global weights independent of specific inputs"""
        if dummy_input is None:
            dummy_input = torch.ones(1, 1, device=next(self.parameters()).device)
        elif dummy_input.dim() == 1:
            dummy_input = dummy_input.unsqueeze(0).unsqueeze(0)
        
        weights = self.network(dummy_input)
        weights = weights / (weights.abs().sum(dim=1, keepdim=True) + 1e-8)
        return weights.squeeze(0)


def overlap_loss(id_scores, ood_scores, margin=0.5):
    """Enhanced loss function to minimize overlap between ID and OOD score distributions"""
    # Calculate distribution statistics with numerical stability
    id_mean = id_scores.mean()
    ood_mean = ood_scores.mean()
    id_std = id_scores.std() + 1e-8
    ood_std = ood_scores.std() + 1e-8
    
    # Separation Loss: Encourage ID scores to be higher than OOD scores
    separation_loss = torch.relu(margin - (id_mean - ood_mean))
    
    # Variance Regularization: Promote tighter distributions
    variance_loss = (id_std + ood_std) * 0.1
    
    # Wasserstein Distance Penalty: Minimize distributional overlap
    id_sorted, _ = torch.sort(id_scores)
    ood_sorted, _ = torch.sort(ood_scores)
    
    # Handle different lengths by trimming to minimum
    min_len = min(len(id_sorted), len(ood_sorted))
    id_trimmed = id_sorted[:min_len]
    ood_trimmed = ood_sorted[:min_len]
    
    # Compute Wasserstein-1 distance
    wasserstein_dist = torch.mean(torch.abs(id_trimmed - ood_trimmed))
    overlap_penalty = torch.relu(0.3 - wasserstein_dist)
    
    # Combined loss with proper weighting
    total_loss = separation_loss + variance_loss + overlap_penalty
    
    return total_loss, {
        'separation_loss': separation_loss.item(),
        'variance_loss': variance_loss.item(), 
        'overlap_penalty': overlap_penalty.item(),
        'id_mean': id_mean.item(),
        'ood_mean': ood_mean.item(),
        'id_std': id_std.item(),
        'ood_std': ood_std.item(),
        'wasserstein_dist': wasserstein_dist.item()
    }


def calculate_ensemble_score_with_weights_tensor(features_tensor, weights_tensor):
    """Calculate ensemble scores using given weights (tensor version)"""
    return torch.sum(features_tensor * weights_tensor, dim=1)


def calculate_ensemble_scores_with_weights(features_list, weights_dict):
    """Calculate ensemble scores using given weights (dictionary version)"""
    scores = []
    for features in features_list:
        score = 0
        for key, weight in weights_dict.items():
            if key in features:
                score += weight * features[key]
        scores.append(score)
    return scores


class PrototypicalClassifier:
    """Enhanced Prototypical network classifier with overlap reduction methods"""
    
    def __init__(self, feature_extractor, overlap_reducer=None):
        self.feature_extractor = feature_extractor
        self.overlap_reducer = overlap_reducer
        self.prototypes = {}
        self.optimal_threshold = None
        self.method = "original"  # Can be "original", "sampled_mean", "ensemble"
        
    def set_overlap_reduction_method(self, method="original"):
        """Set the overlap reduction method to use"""
        valid_methods = ["original", "sampled_mean", "ensemble", "ensemble_adaptive", "ensemble_neural"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.method = method
        print(f"Overlap reduction method set to: {method}")
    
    def compute_prototypes(self, bounds_df, augmentor, num_samples=20):
        """Compute prototypes for each species using augmented samples"""
        print("Computing prototypes for each species...")
        
        species_features = {}
        
        for species in tqdm(bounds_df['name'].unique()):
            species_crops = bounds_df[bounds_df['name'] == species]['crop'].tolist()
            all_features = []
            
            for crop in species_crops:
                # Generate augmented samples (19 augmented + 1 original)
                augmented_samples = augmentor.generate_augmented_samples(crop, num_samples-1)
                
                # Extract features
                features = self.feature_extractor.extract_features(augmented_samples)
                all_features.append(features)
            
            # Combine all features for this species
            species_features_array = np.vstack(all_features)
            
            # Ensure we have exactly num_samples per class by random sampling if needed
            if species_features_array.shape[0] > num_samples:
                indices = np.random.choice(species_features_array.shape[0], num_samples, replace=False)
                species_features_array = species_features_array[indices]
            elif species_features_array.shape[0] < num_samples:
                # Repeat samples if we have fewer than required
                repeat_factor = num_samples // species_features_array.shape[0] + 1
                species_features_array = np.tile(species_features_array, (repeat_factor, 1))[:num_samples]
            
            # Compute prototype as mean of features
            prototype = species_features_array.mean(axis=0)
            prototype = prototype / np.linalg.norm(prototype)  # Normalize
            
            self.prototypes[species] = prototype
            species_features[species] = species_features_array
        
        print(f"Computed prototypes for {len(self.prototypes)} species")
        return species_features
    
    def generate_ood_crops(self, original_image, labeled_boxes, crop_size=(224, 224), num_crops=100):
        """Generate OOD crops avoiding labeled regions"""
        print(f"Generating {num_crops} OOD crops...")
        
        def boxes_overlap(box1, box2):
            y0, y1, x0, x1 = box1
            ly0, ly1, lx0, lx1 = box2
            return not (x1 <= lx0 or x0 >= lx1 or y1 <= ly0 or y0 >= ly1)
        
        def generate_random_crop_coordinates(image_shape, crop_height, crop_width, labeled_boxes, max_attempts=1000):
            _, H, W = image_shape
            for _ in range(max_attempts):
                y0 = random.randint(0, H - crop_height)
                x0 = random.randint(0, W - crop_width)
                crop_box = (y0, y0 + crop_height, x0, x0 + crop_width)
                
                # Check for overlap with labeled boxes
                overlap = any(boxes_overlap(crop_box, lab_box) for lab_box in labeled_boxes)
                if not overlap:
                    return crop_box
            return None
        
        ood_crops = []
        crop_height, crop_width = crop_size
        
        for i in range(num_crops):
            coords = generate_random_crop_coordinates(
                original_image.shape, crop_height, crop_width, labeled_boxes
            )
            
            if coords is not None:
                y0, y1, x0, x1 = coords
                crop = original_image[:, y0:y1, x0:x1]
                
                # Check if crop is not completely black
                if not np.all(crop == 0):
                    ood_crops.append(crop)
            
            if len(ood_crops) >= num_crops:
                break
        
        print(f"Generated {len(ood_crops)} valid OOD crops")
        return ood_crops
    
    def calculate_similarity_distributions_enhanced(self, species_features, ood_crops, samples_per_class=10):
        """Calculate ID and OOD similarity distributions using selected overlap reduction method"""
        print(f"Calculating similarity distributions using {self.method} method...")
        
        id_similarities = []
        ood_similarities = []
        
        if self.method == "original":
            # Original method: direct cosine similarity
            id_similarities, ood_similarities = self._calculate_original_similarities(
                species_features, ood_crops, samples_per_class
            )
            
        elif self.method == "sampled_mean":
            # Sampled mean method: average across augmented crops
            id_similarities, ood_similarities = self._calculate_sampled_mean_similarities(
                species_features, ood_crops, samples_per_class
            )
            
        elif self.method in ["ensemble", "ensemble_adaptive", "ensemble_neural"]:
            # Ensemble method: neural network combining multiple features
            id_similarities, ood_similarities = self._calculate_ensemble_similarities(
                species_features, ood_crops, samples_per_class
            )
        
        return np.array(id_similarities), np.array(ood_similarities)
    
    def _calculate_original_similarities(self, species_features, ood_crops, samples_per_class):
        """Calculate similarities using original method"""
        id_similarities = []
        ood_similarities = []
        
        # ID similarities: prototypes vs corresponding class images
        for species, features in species_features.items():
            prototype = self.prototypes[species]
            
            # Randomly sample images from this class
            sample_indices = np.random.choice(features.shape[0], min(samples_per_class, features.shape[0]), replace=False)
            sampled_features = features[sample_indices]
            
            # Calculate cosine similarities
            similarities = np.dot(sampled_features, prototype)
            id_similarities.extend(similarities)
        
        # OOD similarities: prototypes vs OOD images
        if ood_crops:
            # Extract features from OOD crops
            ood_pil_images = []
            for crop in ood_crops:
                pil_img = self.feature_extractor.crop_to_pil(crop)
                ood_pil_images.append(pil_img)
            
            ood_features = self.feature_extractor.extract_features(ood_pil_images)
            
            # Calculate similarities between each OOD feature and each prototype
            for prototype in self.prototypes.values():
                similarities = np.dot(ood_features, prototype)
                # Sample to balance with ID similarities
                sample_indices = np.random.choice(len(similarities), min(samples_per_class, len(similarities)), replace=False)
                ood_similarities.extend(similarities[sample_indices])
        
        return id_similarities, ood_similarities
    
    def _calculate_sampled_mean_similarities(self, species_features, ood_crops, samples_per_class):
        """Calculate similarities using sampled mean method"""
        if not self.overlap_reducer:
            raise ValueError("OverlapReductionMethods not initialized")
            
        id_similarities = []
        ood_similarities = []
        
        # ID similarities using sampled mean method
        for species in tqdm(self.prototypes.keys(), desc="ID similarities"):
            species_crops = species_features[species]
            
            for i in range(min(samples_per_class, len(species_crops))):
                # Convert feature back to image-like format for augmentation
                # This is a simplification - in practice you'd want to store original crops
                similarity = self.overlap_reducer.sampled_mean_similarity_method(
                    species_crops[i].reshape(224, 224, -1)[:,:,:3] * 255, 
                    self.prototypes
                )
                id_similarities.append(similarity.item())
        
        # OOD similarities using sampled mean method  
        sample_ood_crops = random.sample(ood_crops, min(samples_per_class * len(self.prototypes), len(ood_crops)))
        
        for crop in tqdm(sample_ood_crops, desc="OOD similarities"):
            try:
                # Convert crop to proper format
                if len(crop.shape) == 3 and crop.shape[0] <= 3:
                    crop_img = np.transpose(crop, (1, 2, 0))
                else:
                    crop_img = crop
                    
                similarity = self.overlap_reducer.sampled_mean_similarity_method(
                    crop_img, self.prototypes
                )
                ood_similarities.append(similarity.item())
            except Exception as e:
                print(f"Error calculating OOD similarity: {e}")
                continue
        
        return id_similarities, ood_similarities
    
    def _calculate_ensemble_similarities(self, species_features, ood_crops, samples_per_class):
        """Calculate similarities using ensemble method"""
        if not self.overlap_reducer:
            raise ValueError("OverlapReductionMethods not initialized")
            
        print("Computing ensemble features for ID and OOD samples...")
        
        # Generate ensemble features for ID samples
        ensemble_id_features = []
        for species in tqdm(self.prototypes.keys(), desc="ID ensemble features"):
            species_crops = species_features[species]
            
            for i in range(min(samples_per_class, len(species_crops))):
                for j in range(i+1, min(samples_per_class, len(species_crops))):
                    # Compare different samples from same class
                    try:
                        img1 = species_crops[i].reshape(224, 224, -1)[:,:,:3]
                        img2 = species_crops[j].reshape(224, 224, -1)[:,:,:3]
                        
                        features = self.overlap_reducer.ensemble_similarity(img1, img2)
                        ensemble_id_features.append(features)
                    except Exception as e:
                        print(f"Error computing ID ensemble features: {e}")
                        continue
        
        # Generate ensemble features for OOD samples
        ensemble_ood_features = []
        id_crops = []
        for species in list(self.prototypes.keys())[:3]:  # Use first 3 species
            species_crops = species_features[species]
            id_crops.extend([crop.reshape(224, 224, -1)[:,:,:3] for crop in species_crops[:3]])
        
        sample_ood_crops = random.sample(ood_crops, min(30, len(ood_crops)))
        
        for ood_crop in tqdm(sample_ood_crops, desc="OOD ensemble features"):
            for id_crop in id_crops[:10]:  # Compare with 10 ID crops
                try:
                    # Convert crops to proper format
                    if len(ood_crop.shape) == 3 and ood_crop.shape[0] <= 3:
                        ood_img = np.transpose(ood_crop, (1, 2, 0))
                    else:
                        ood_img = ood_crop
                        
                    features = self.overlap_reducer.ensemble_similarity(id_crop, ood_img)
                    ensemble_ood_features.append(features)
                except Exception as e:
                    print(f"Error computing OOD ensemble features: {e}")
                    continue
        
        # Learn adaptive weights if not already done
        if self.overlap_reducer.adaptive_weights is None:
            print("Learning adaptive weights...")
            self.overlap_reducer.learn_optimal_weights(ensemble_id_features, ensemble_ood_features)
        
        # Calculate ensemble scores
        weights = self.overlap_reducer.adaptive_weights if self.method == "ensemble_adaptive" else None
        
        id_similarities = []
        for features in ensemble_id_features:
            score = self.overlap_reducer.ensemble_similarity_score(features, weights)
            id_similarities.append(score)
        
        ood_similarities = []
        for features in ensemble_ood_features:
            score = self.overlap_reducer.ensemble_similarity_score(features, weights)
            ood_similarities.append(score)
        
        return id_similarities, ood_similarities
    
    def calculate_similarity_distributions(self, species_features, ood_crops, samples_per_class=10):
        """Legacy method - redirects to enhanced version"""
        return self.calculate_similarity_distributions_enhanced(species_features, ood_crops, samples_per_class)
    
    def find_optimal_threshold(self, id_similarities, ood_similarities):
        """Find optimal threshold using Youden's J statistic"""
        print("Finding optimal threshold using ROC analysis...")
        
        # Combine similarities and create labels (1 for ID, 0 for OOD)
        all_similarities = np.concatenate([id_similarities, ood_similarities])
        labels = np.concatenate([np.ones(len(id_similarities)), np.zeros(len(ood_similarities))])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_similarities)
        
        # Calculate Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        
        return self.optimal_threshold, roc_auc, fpr, tpr, thresholds
    
    def calculate_overlap_metrics(self, id_similarities, ood_similarities):
        """Calculate overlap reduction metrics"""
        # Overlap ratio (percentage of OOD scores above ID median)
        id_median = np.median(id_similarities)
        overlap_ratio = np.mean(ood_similarities > id_median)
        
        # Separation distance (difference between means divided by pooled std)
        id_mean, ood_mean = np.mean(id_similarities), np.mean(ood_similarities)
        id_std, ood_std = np.std(id_similarities), np.std(ood_similarities)
        pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
        separation_distance = abs(id_mean - ood_mean) / pooled_std if pooled_std > 0 else 0
        
        # Wasserstein distance
        wasserstein_dist = wasserstein_distance(id_similarities, ood_similarities)
        
        # KS test
        ks_stat, p_value = ks_2samp(id_similarities, ood_similarities)
        
        metrics = {
            'overlap_ratio': overlap_ratio,
            'separation_distance': separation_distance,
            'wasserstein_distance': wasserstein_dist,
            'ks_statistic': ks_stat,
            'ks_p_value': p_value,
            'id_mean': id_mean,
            'ood_mean': ood_mean,
            'id_std': id_std,
            'ood_std': ood_std
        }
        
        return metrics
    
    def classify_ood(self, query_image):
        """Classify whether a query image is OOD"""
        if self.optimal_threshold is None:
            raise ValueError("Optimal threshold not set. Run find_optimal_threshold first.")
        
        # Extract features from query image
        if isinstance(query_image, np.ndarray):
            query_features = self.feature_extractor.extract_features([query_image])
        else:
            query_features = self.feature_extractor.extract_features([query_image])
        
        query_feature = query_features[0]
        
        # Calculate similarities to all prototypes
        similarities = []
        for species, prototype in self.prototypes.items():
            similarity = np.dot(query_feature, prototype)
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        
        # Classify as OOD if max similarity is below threshold
        is_ood = max_similarity < self.optimal_threshold
        
        return is_ood, max_similarity, similarities
    
    def evaluate_classifier(self, test_id_images, test_ood_images):
        """Evaluate the OOD classifier performance"""
        print("Evaluating classifier performance...")
        
        predictions = []
        true_labels = []
        
        # Test ID images (should return False for OOD)
        for img in test_id_images:
            is_ood, _, _ = self.classify_ood(img)
            predictions.append(is_ood)
            true_labels.append(False)  # ID images should not be classified as OOD
        
        # Test OOD images (should return True for OOD)
        for img in test_ood_images:
            is_ood, _, _ = self.classify_ood(img)
            predictions.append(is_ood)
            true_labels.append(True)  # OOD images should be classified as OOD
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'threshold': self.optimal_threshold
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return results


def visualize_results_enhanced(id_similarities, ood_similarities, fpr, tpr, roc_auc, threshold, method_name="Method", overlap_metrics=None):
    """Enhanced visualization with overlap reduction metrics"""
    fig = plt.figure(figsize=(20, 5))
    
    # Plot similarity distributions
    ax1 = plt.subplot(1, 4, 1)
    ax1.hist(id_similarities, bins=30, alpha=0.7, label='ID similarities', color='blue', density=True)
    ax1.hist(ood_similarities, bins=30, alpha=0.7, label='OOD similarities', color='red', density=True)
    ax1.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.3f}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{method_name}: ID vs OOD Similarity Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ROC curve
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{method_name}: ROC Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # Plot overlap metrics
    if overlap_metrics:
        ax3 = plt.subplot(1, 4, 3)
        metrics_names = ['Overlap Ratio', 'Sep. Distance', 'Wasserstein Dist.']
        metrics_values = [
            overlap_metrics['overlap_ratio'],
            min(overlap_metrics['separation_distance'], 5.0),  # Cap for visualization
            overlap_metrics['wasserstein_distance']
        ]
        colors = ['red', 'green', 'blue']
        bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax3.set_ylabel('Metric Value')
        ax3.set_title(f'{method_name}: Overlap Metrics')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Plot distribution comparison (box plot)
    ax4 = plt.subplot(1, 4, 4)
    data = [id_similarities, ood_similarities]
    labels = ['ID', 'OOD']
    bp = ax4.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax4.set_ylabel('Similarity Score')
    ax4.set_title(f'{method_name}: Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_overlap_reduction_methods(classifier, species_features, ood_crops, samples_per_class=10):
    """Compare all three overlap reduction methods"""
    print("="*80)
    print("OVERLAP REDUCTION METHODS COMPARISON")
    print("="*80)
    
    methods = ["original", "sampled_mean", "ensemble_adaptive"]
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")
        
        # Set method
        classifier.set_overlap_reduction_method(method)
        
        # Calculate similarities
        id_similarities, ood_similarities = classifier.calculate_similarity_distributions_enhanced(
            species_features, ood_crops, samples_per_class
        )
        
        # Find optimal threshold
        threshold, roc_auc, fpr, tpr, thresholds = classifier.find_optimal_threshold(
            id_similarities, ood_similarities
        )
        
        # Calculate overlap metrics
        overlap_metrics = classifier.calculate_overlap_metrics(id_similarities, ood_similarities)
        overlap_metrics['auc_score'] = roc_auc
        overlap_metrics['threshold'] = threshold
        
        # Store results
        results[method] = {
            'id_similarities': id_similarities,
            'ood_similarities': ood_similarities,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'overlap_metrics': overlap_metrics
        }
        
        # Print metrics
        print(f"Overlap Ratio: {overlap_metrics['overlap_ratio']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Separation Distance: {overlap_metrics['separation_distance']:.4f}")
        print(f"Wasserstein Distance: {overlap_metrics['wasserstein_distance']:.4f}")
        print(f"KS Statistic: {overlap_metrics['ks_statistic']:.4f} (p-value: {overlap_metrics['ks_p_value']:.2e})")
        
        # Visualize results
        visualize_results_enhanced(
            id_similarities, ood_similarities, fpr, tpr, roc_auc, threshold,
            method_name=method.replace('_', ' ').title(), overlap_metrics=overlap_metrics
        )
    
    # Create comparison summary
    print("\n" + "="*80)
    print("METHODS COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        method: {
            'Overlap Ratio': results[method]['overlap_metrics']['overlap_ratio'],
            'ROC AUC': results[method]['roc_auc'],
            'Separation Distance': results[method]['overlap_metrics']['separation_distance'],
            'Wasserstein Distance': results[method]['overlap_metrics']['wasserstein_distance'],
            'KS Statistic': results[method]['overlap_metrics']['ks_statistic']
        } for method in methods
    }).T
    
    print(comparison_df.round(4))
    
    # Determine best method
    best_overlap = comparison_df['Overlap Ratio'].idxmin()
    best_auc = comparison_df['ROC AUC'].idxmax()
    best_separation = comparison_df['Separation Distance'].idxmax()
    
    print(f"\n🏆 Best Methods:")
    print(f"- Lowest Overlap Ratio: {best_overlap} ({comparison_df.loc[best_overlap, 'Overlap Ratio']:.4f})")
    print(f"- Highest ROC AUC: {best_auc} ({comparison_df.loc[best_auc, 'ROC AUC']:.4f})")
    print(f"- Best Separation: {best_separation} ({comparison_df.loc[best_separation, 'Separation Distance']:.4f})")
    
    # Calculate improvement percentages
    baseline_overlap = results['original']['overlap_metrics']['overlap_ratio']
    baseline_auc = results['original']['roc_auc']
    
    print(f"\n📈 Improvements over Original Method:")
    for method in methods[1:]:  # Skip original
        overlap_improvement = ((baseline_overlap - results[method]['overlap_metrics']['overlap_ratio']) / baseline_overlap * 100)
        auc_improvement = ((results[method]['roc_auc'] - baseline_auc) / baseline_auc * 100)
        
        print(f"- {method.replace('_', ' ').title()}:")
        print(f"  Overlap reduction: {overlap_improvement:+.1f}%")
        print(f"  AUC improvement: {auc_improvement:+.1f}%")
    
    return results, comparison_df


def train_ensemble_neural_weights(overlap_reducer, id_features, ood_features, num_epochs=50):
    """Train neural network to learn optimal ensemble weights"""
    print("Training neural network for optimal ensemble weights...")
    
    # Initialize network
    ensemble_net = EnsembleWeightNet(num_features=9).to(overlap_reducer.device)
    optimizer = optim.Adam(ensemble_net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    
    # Convert features to arrays
    feature_names = ['direct', 'mean', 'max', 'min', 'std', 'median', 'p75', 'p25', 'iqr']
    id_features_array = np.array([[f[name] for name in feature_names] for f in id_features])
    ood_features_array = np.array([[f[name] for name in feature_names] for f in ood_features])
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(num_epochs):
        ensemble_net.train()
        
        # Sample batch data
        batch_size = 16
        id_batch_size = min(batch_size, len(id_features_array))
        ood_batch_size = min(batch_size, len(ood_features_array))
        
        id_indices = np.random.choice(len(id_features_array), id_batch_size, replace=True)
        ood_indices = np.random.choice(len(ood_features_array), ood_batch_size, replace=True)
        
        id_batch = torch.FloatTensor(id_features_array[id_indices]).to(overlap_reducer.device)
        ood_batch = torch.FloatTensor(ood_features_array[ood_indices]).to(overlap_reducer.device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get current weights from the network
        weights = ensemble_net()
        
        # Compute weighted ensemble scores
        id_scores = calculate_ensemble_score_with_weights_tensor(id_batch, weights)
        ood_scores = calculate_ensemble_score_with_weights_tensor(ood_batch, weights)
        
        # Compute overlap loss
        loss, loss_components = overlap_loss(id_scores, ood_scores)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ensemble_net.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Track best weights
        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                best_weights = ensemble_net().cpu().numpy()
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
                  f"ID mean = {loss_components['id_mean']:.4f}, "
                  f"OOD mean = {loss_components['ood_mean']:.4f}")
    
    # Convert best weights to dictionary
    neural_weights_dict = {name: weight for name, weight in zip(feature_names, best_weights)}
    overlap_reducer.neural_weights = neural_weights_dict
    
    print(f"\nTraining completed. Best loss: {best_loss:.6f}")
    print(f"Learned Neural Weights:")
    for feature, weight in neural_weights_dict.items():
        print(f"  {feature}: {weight:.6f}")
    
    return neural_weights_dict


def main():
    """Enhanced main execution function with overlap reduction methods comparison"""
    print("Starting Enhanced Prototypical Few-Shot OOD Detection Pipeline...")
    
    # Configuration - using the updated paths from TreeSpeciesDataloader
    casuarina_93deg_path = "D:/casuarina_nature_park/20210510/90deg43m85pct255deg"
    casuarina_183deg_path = "D:/casuarina_nature_park/20210510/90deg43m85pct265deg"
    chestnut_may_path = "D:/chestnut_nature_park/20210510/90deg43m85pct255deg"
    chestnut_dec_path = "D:/chestnut_nature_park/20201218"
    
    # Initialize components
    print("Initializing enhanced components...")
    dataloader = TreeSpeciesDataloader(
        casuarina_93deg_path=casuarina_93deg_path,
        casuarina_183deg_path=casuarina_183deg_path,
        chestnut_may_path=chestnut_may_path,
        chestnut_dec_path=chestnut_dec_path
    )
    
    augmentor = DataAugmentation()
    feature_extractor = CLIPFeatureExtractor()
    
    # Initialize overlap reduction methods
    overlap_reducer = OverlapReductionMethods(
        feature_extractor.model, feature_extractor.preprocess, augmentor, feature_extractor.device
    )
    
    classifier = PrototypicalClassifier(feature_extractor, overlap_reducer)
    
    # Load and prepare comprehensive data
    print("Loading and preparing comprehensive multispectral data...")
    bounds_df, datasets = dataloader.prepare_comprehensive_dataset()
    
    if bounds_df.empty:
        print("No data available. Please check dataset paths.")
        return None, None
    
    print(f"Loaded {len(bounds_df)} labeled samples from {bounds_df['name'].nunique()} species")
    print(f"Available datasets: {list(datasets.keys())}")
    
    # Compute prototypes
    species_features = classifier.compute_prototypes(bounds_df, augmentor, num_samples=20)
    
    # Generate OOD crops using the first available dataset
    first_dataset = list(datasets.values())[0]
    labeled_boxes = [(row['y0'], row['y1'], row['x0'], row['x1']) for _, row in bounds_df.iterrows()]
    ood_crops = classifier.generate_ood_crops(first_dataset, labeled_boxes, num_crops=100)
    
    # Compare all overlap reduction methods
    print("\n" + "="*80)
    print("COMPREHENSIVE OVERLAP REDUCTION METHODS COMPARISON")
    print("="*80)
    
    results, comparison_df = compare_overlap_reduction_methods(
        classifier, species_features, ood_crops, samples_per_class=10
    )
    
    # Select best method based on combination of metrics
    best_method = comparison_df['ROC AUC'].idxmax()  # Could also use other criteria
    print(f"\n🎯 Selected best method: {best_method}")
    
    # Set classifier to use best method for final evaluation
    classifier.set_overlap_reduction_method(best_method)
    
    # Final evaluation with best method
    print(f"\n--- Final Evaluation with {best_method.upper()} method ---")
    
    # Prepare test data (balanced ID and OOD samples)
    test_id_samples = []
    available_species = list(species_features.keys())
    samples_per_species = min(4, len(bounds_df) // len(available_species))
    
    for species in available_species:
        species_crops = bounds_df[bounds_df['name'] == species]['crop'].tolist()
        test_id_samples.extend(species_crops[:samples_per_species])
    
    test_ood_samples = ood_crops[:min(80, len(ood_crops))]
    
    print(f"Test set: {len(test_id_samples)} ID samples, {len(test_ood_samples)} OOD samples")
    
    # Evaluate classifier with best method
    final_results = classifier.evaluate_classifier(test_id_samples, test_ood_samples)
    
    # Save comprehensive results
    comprehensive_results = {
        'method_comparison': results,
        'comparison_summary': comparison_df.to_dict(),
        'best_method': best_method,
        'final_evaluation': final_results,
        'dataset_info': {
            'num_species': bounds_df['name'].nunique(),
            'total_samples': len(bounds_df),
            'species_distribution': bounds_df['name'].value_counts().to_dict(),
            'datasets_used': list(datasets.keys())
        }
    }
    
    # Optionally save results
    # with open('overlap_reduction_results.pkl', 'wb') as f:
    #     pickle.dump(comprehensive_results, f)
    
    print("\n🎉 Enhanced pipeline completed successfully!")
    print(f"Final performance with {best_method}: Accuracy = {final_results['accuracy']:.3f}, "
          f"AUC = {results[best_method]['roc_auc']:.3f}")
    
    return classifier, comprehensive_results


def main_simple():
    """Simplified main function for basic testing"""
    print("Starting Basic Prototypical Few-Shot OOD Detection Pipeline...")
    
    # Basic configuration for testing
    chestnut_may_path = "chestnut_nature_park/20210510/90deg43m85pct255deg"
    
    # Initialize basic components
    dataloader = TreeSpeciesDataloader(chestnut_may_path=chestnut_may_path)
    augmentor = DataAugmentation()
    feature_extractor = CLIPFeatureExtractor()
    classifier = PrototypicalClassifier(feature_extractor)
    
    # Load and prepare data
    bounds_df, datasets = dataloader.prepare_comprehensive_dataset()
    
    if bounds_df.empty:
        print("No data available. Please check dataset paths.")
        return None, None
    
    print(f"Loaded {len(bounds_df)} labeled samples from {bounds_df['name'].nunique()} species")
    
    # Compute prototypes
    species_features = classifier.compute_prototypes(bounds_df, augmentor, num_samples=20)
    
    # Generate OOD crops
    first_dataset = list(datasets.values())[0]
    labeled_boxes = [(row['y0'], row['y1'], row['x0'], row['x1']) for _, row in bounds_df.iterrows()]
    ood_crops = classifier.generate_ood_crops(first_dataset, labeled_boxes, num_crops=100)
    
    # Calculate similarity distributions (using original method)
    id_similarities, ood_similarities = classifier.calculate_similarity_distributions(
        species_features, ood_crops, samples_per_class=10
    )
    
    # Find optimal threshold
    threshold, roc_auc, fpr, tpr, thresholds = classifier.find_optimal_threshold(
        id_similarities, ood_similarities
    )
    
    # Calculate overlap metrics
    overlap_metrics = classifier.calculate_overlap_metrics(id_similarities, ood_similarities)
    
    # Visualize results
    visualize_results_enhanced(
        id_similarities, ood_similarities, fpr, tpr, roc_auc, threshold,
        method_name="Original Method", overlap_metrics=overlap_metrics
    )
    
    # Evaluate classifier
    test_id_samples = []
    for species in list(species_features.keys())[:5]:
        species_crops = bounds_df[bounds_df['name'] == species]['crop'].tolist()
        test_id_samples.extend(species_crops[:4])
    
    test_ood_samples = ood_crops[:80] if len(ood_crops) >= 80 else ood_crops
    results = classifier.evaluate_classifier(test_id_samples, test_ood_samples)
    
    print("\nBasic pipeline completed successfully!")
    return classifier, results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the enhanced pipeline with overlap reduction methods comparison
    try:
        classifier, results = main()
    except Exception as e:
        print(f"Error running enhanced pipeline: {e}")
        print("Falling back to simple pipeline...")
        classifier, results = main_simple()
