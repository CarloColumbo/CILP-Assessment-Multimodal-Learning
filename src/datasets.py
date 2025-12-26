import torch
import torchvision.transforms as transforms
from PIL import Image

    
class CustomTorchImageDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading RGB and LiDAR modalities from a FiftyOne dataset.
    RGB images are loaded from file paths, while LiDAR data is assumed to be preprocessed
    and stored as numpy arrays in the FiftyOne dataset.
    
    Args:
        fiftyone_dataset: FiftyOne dataset instance
        img_size: Desired image size (height, width) for resizing
        device: Torch device to load tensors onto
        image_transforms: Optional torchvision transforms to apply to images
        label_map: Optional dictionary mapping string labels to integer indices
        gt_field: Field name in FiftyOne dataset containing ground truth labels
        
    Returns:
        PyTorch Dataset yielding (image, lidar_xyza, label_idx) tuples
    """
    def __init__(self,
                 fiftyone_dataset,
                 img_size,
                 device,
                 image_transforms=None,
                 label_map=None,
                 gt_field="ground_truth"):
        self.fiftyone_dataset = fiftyone_dataset
        self.device = device
        self.rgb_paths = self.fiftyone_dataset.select_group_slices("rgb").values("filepath")
        self.lidar_xyzas = self.fiftyone_dataset.select_group_slices("lidar_img").values("xyza")
        
        assert len(self.rgb_paths) == len(self.lidar_xyzas), "Mismatch between number of RGB and LiDAR samples"
        
        self.str_labels = self.fiftyone_dataset.values(f"{gt_field}.label")

        self.image_transforms = image_transforms or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # Scales data into [0,1]
        ])

        if label_map is None:
            labels = set(self.str_labels)
            self.label_map = {label: idx for idx, label in enumerate(sorted(labels))}
        else:
            self.label_map = label_map

        print(f"CustomTorchImageDataset initialized with {len(self.rgb_paths)} samples.")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        lidar_xyza = self.lidar_xyzas[idx]
        
        try:
            rgb_image = Image.open(rgb_path)
        except Exception as e:
            print(f"Error loading image {rgb_path}: {e}")
            
        # Apply image transformations only for RGB image
        if self.image_transforms:
            rgb_image = self.image_transforms(rgb_image).to(self.device)
        
        lidar_xyza = torch.from_numpy(lidar_xyza).to(torch.float32).to(self.device)

        label_str = self.str_labels[idx]
        label_idx = self.label_map.get(label_str, -1)
        if label_idx == -1:
            print(f"Warning: Label '{label_str}' not in label_map for image {rgb_path}")

        label_idx = torch.tensor(label_idx, dtype=torch.float32)[None].to(self.device)

        return rgb_image, lidar_xyza, label_idx
