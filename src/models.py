import torch
import torch.nn as nn
import torch.nn.functional as Fun
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models to make logging simpler.
    
    Args:
        None
        
    Returns:
        None
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_embedding_size(self) -> int:
        """
        Get the size of the embedding produced by the model.
        """
        pass
    
    @abstractmethod
    def get_fusion_strategy(self) -> str:
        """
        Get the fusion strategy used by the model.
        """
        pass
    
    def get_number_of_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        We also count non-trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())


# Task 2
class ConvEncoder(nn.Module):
    """
    Conv encoder that outputs a 100-dim embedding
    """
    def __init__(self, in_ch, embedding_dim=100):
        super().__init__()
        kernel_size = 3

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 8 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, embedding_dim)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return Fun.normalize(emb)


class LateFusionModel(BaseModel):
    """
    LateFusionModel is a model that performs late fusion by combining the outputs
    of two separate encoders for RGB and XYZ inputs.

    Args:
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(self, rgb_in_ch, lidar_in_ch, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        self.rgb_encoder = ConvEncoder(rgb_in_ch, embedding_dim=100)
        self.lidar_encoder = ConvEncoder(lidar_in_ch, embedding_dim=100)

        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)
        fused = torch.cat([rgb_emb, lidar_emb], dim=1)
        logits = self.classifier(fused)
        return logits
    
    def get_embedding_size(self) -> int:
        return self.num_classes
    
    def get_fusion_strategy(self) -> str:
        return "late"

    
class ConcatIntermediateNet(BaseModel):
    """
    ConcatIntermediateNet is a model that performs intermediate fusion by concatenating
    the feature maps from the RGB and XYZ branches.
    
    Args:
        rgb_ch (int): Number of input channels for the RGB images
        xyz_ch (int): Number of input channels for the XYZ images
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(self, rgb_ch, xyz_ch, num_classes=1):
        kernel_size = 3
        super().__init__()
        self.num_classes = num_classes

        self.rgb_path = nn.Sequential(
            nn.Conv2d(rgb_ch, 50, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.lidar_path = nn.Sequential(
            nn.Conv2d(xyz_ch, 50, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        shared_in_ch = 100 + 100
        
        self.shared = nn.Sequential(
            nn.Conv2d(shared_in_ch, 200, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(200, 200, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 8 * 8, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x_rgb, x_xyz):
        rgb_feat = self.rgb_path(x_rgb)
        lidar_feat = self.lidar_path(x_xyz)

        x = torch.cat([rgb_feat, lidar_feat], dim=1)

        x = self.shared(x)
        logits = self.classifier(x)
        return logits
    
    def get_embedding_size(self) -> int:
        return self.num_classes
    
    def get_fusion_strategy(self) -> str:
        return "intermediate"
    
    
class AdditionIntermediateNet(BaseModel):
    """
    AdditionIntermediateNet is a model that performs intermediate fusion by adding
    the feature maps from the RGB and XYZ branches element-wise.
    
    Args:
        rgb_ch (int): Number of input channels for the RGB images
        xyz_ch (int): Number of input channels for the XYZ images
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(self, rgb_ch, xyz_ch, num_classes=1):
        kernel_size = 3
        super().__init__()
        self.num_classes = num_classes
        
        self.rgb_path = nn.Sequential(
            nn.Conv2d(rgb_ch, 50, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.lidar_path = nn.Sequential(
            nn.Conv2d(xyz_ch, 50, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        shared_in_ch = 100
        
        self.shared = nn.Sequential(
            nn.Conv2d(shared_in_ch, 200, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 8 * 8, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x_rgb, x_xyz):
        rgb_feat = self.rgb_path(x_rgb)
        lidar_feat = self.lidar_path(x_xyz)

        x = rgb_feat + lidar_feat

        x = self.shared(x)
        logits = self.classifier(x)
        return logits
    
    def get_embedding_size(self) -> int:
        return self.num_classes
    
    def get_fusion_strategy(self) -> str:
        return "intermediate"


class HadamardIntermediateNet(BaseModel):
    """
    HadamardIntermediateNet is a model that performs intermediate fusion by multiplying
    the feature maps from the RGB and XYZ branches element-wise.
    
    Args:
        rgb_ch (int): Number of input channels for the RGB images
        xyz_ch (int): Number of input channels for the XYZ images
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(self, rgb_ch, xyz_ch, num_classes=1):
        kernel_size = 3
        super().__init__()
        self.num_classes = num_classes
        
        self.rgb_path = nn.Sequential(
            nn.Conv2d(rgb_ch, 50, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.lidar_path = nn.Sequential(
            nn.Conv2d(xyz_ch, 50, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        shared_in_ch = 100
        
        self.shared = nn.Sequential(
            nn.Conv2d(shared_in_ch, 200, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 8 * 8, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x_rgb, x_xyz):
        rgb_feat = self.rgb_path(x_rgb)
        lidar_feat = self.lidar_path(x_xyz)

        x = rgb_feat * lidar_feat

        x = self.shared(x)
        logits = self.classifier(x)
        return logits
    
    def get_embedding_size(self) -> int:
        return self.num_classes
    
    def get_fusion_strategy(self) -> str:
        return "intermediate"


# Task 3
class ConcatIntermediateNetWithStride(BaseModel):
    """
    ConcatIntermediateNetWithStride is a model that performs intermediate fusion by concatenating
    the feature maps from the RGB and XYZ branches.
    
    Args:
        rgb_ch (int): Number of input channels for the RGB images
        xyz_ch (int): Number of input channels for the XYZ images
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(self, rgb_ch, xyz_ch, num_classes=1):
        kernel_size = 3
        super().__init__()
        self.num_classes = num_classes

        self.rgb_path = nn.Sequential(
            nn.Conv2d(rgb_ch, 50, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(50, 100, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.lidar_path = nn.Sequential(
            nn.Conv2d(xyz_ch, 50, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(50, 100, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        
        shared_in_ch = 100 + 100
        
        self.shared = nn.Sequential(
            nn.Conv2d(shared_in_ch, 200, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(200, 200, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 8 * 8, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x_rgb, x_xyz):
        rgb_feat = self.rgb_path(x_rgb)
        lidar_feat = self.lidar_path(x_xyz)

        x = torch.cat([rgb_feat, lidar_feat], dim=1)

        x = self.shared(x)
        logits = self.classifier(x)
        return logits
    
    def get_embedding_size(self) -> int:
        return self.num_classes
    
    def get_fusion_strategy(self) -> str:
        return "intermediate"
    

# Task 4
class Embedder(BaseModel):
    """
    Embedder is a model that produces normalized embeddings from input images.
    It is copied from the assessment.

    Args:
        in_ch (int): Number of input channels
        emb_size (int): Size of the output embedding
        
    Returns:
        Normalized embeddings of size emb_size
    """
    def __init__(self, in_ch, emb_size):
        super().__init__()
        self.embedding_size = emb_size
        kernel_size = 3

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return Fun.normalize(emb)
    
    def get_embedding_size(self) -> int:
        return self.embedding_size
    
    def get_fusion_strategy(self) -> str:
        return "N/A"
    
    
class EmbedderStrided(BaseModel):
    """
    EmbedderStrided is a model that produces normalized embeddings from input images.
    It is a variant of Embedder that uses strided convolutions instead of max pooling.

    Args:
        in_ch (int): Number of input channels
        emb_size (int): Size of the output embedding
        
    Returns:
        Normalized embeddings of size emb_size
    """
    def __init__(self, in_ch, emb_size):
        super().__init__()
        self.embedding_size = emb_size
        kernel_size = 3

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 100, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 200, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return Fun.normalize(emb)
    
    def get_embedding_size(self) -> int:
        return self.embedding_size
    
    def get_fusion_strategy(self) -> str:
        return "N/A"


class ContrastivePretraining(BaseModel):
    """
    ContrastivePretraining is a model that performs contrastive learning
    using cosine similarity between image and LiDAR embeddings.
    It is copied from the assessment.

    Args:
        img_embedder (BaseModel): The embedder for RGB images
        lidar_embedder (BaseModel): The embedder for LiDAR depth maps

    Returns:
        Logits for image and LiDAR embeddings
    """
    def __init__(self, img_embedder: BaseModel, lidar_embedder: BaseModel):
        super().__init__()
        self.img_embedder = img_embedder
        self.lidar_embedder = lidar_embedder
        self.cos = nn.CosineSimilarity()

    def forward(self, rgb_imgs, lidar_depths):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_depths)
        
        batch_size = img_emb.size(0)

        repeated_img_emb = img_emb.repeat_interleave(len(img_emb), dim=0)
        repeated_lidar_emb = lidar_emb.repeat(len(lidar_emb), 1)
        
        similarity = self.cos(repeated_img_emb, repeated_lidar_emb)
        similarity = torch.unflatten(similarity, 0, (batch_size, batch_size))
        similarity = (similarity + 1) / 2
 
        logits_per_img = similarity
        logits_per_lidar = similarity.T
        return logits_per_img, logits_per_lidar
    
    def get_embedding_size(self) -> int:
        return self.img_embedder.get_embedding_size()

    def get_fusion_strategy(self) -> str:
        return "contrastive"


class Projector(BaseModel):
    """
    Projector is a model that projects image embeddings to LiDAR embedding space.
    It is copied from the assessment.

    Args:
        in_emb_size (int): Size of the input embedding
        out_emb_size (int): Size of the output embedding

    Returns:
        Projected embeddings of size out_emb_size
    """
    def __init__(self, in_emb_size, out_emb_size):
        super().__init__()
        self.out_emb_size = out_emb_size
        self.net = nn.Sequential(
            nn.Linear(in_emb_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, out_emb_size)
        )
        
    def forward(self, imgs):
        return self.net(imgs)
    
    def get_embedding_size(self) -> int:
        return self.out_emb_size

    def get_fusion_strategy(self) -> str:
        return "N/A"


class RGB2LiDARClassifier(BaseModel):
    """
    RGB2LiDARClassifier is a model that classifies projected RGB embeddings
    
    Args:
        projector (Projector): The projector model
        img_embedder (Embedder): The embedder for RGB images
        num_classes (int): Number of output classes
        
    Returns:
        Output logits for each class
    """
    def __init__(
            self,
            projector: Projector,
            img_embedder: Embedder,
            num_classes: int = 1
    ):
        super().__init__()
        self.projector = projector
        self.img_embedder = img_embedder
        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, imgs):
        img_encodings = self.img_embedder(imgs)
        proj_lidar_embs = self.projector(img_encodings)
        return self.classifier(proj_lidar_embs)
    
    def get_embedding_size(self) -> int:
        return self.num_classes

    def get_fusion_strategy(self) -> str:
        return "N/A"
