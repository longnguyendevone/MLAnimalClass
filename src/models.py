import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import timm
import torchvision.models as models

#  OPTIMIZED CUSTOM CNN (DSC + Residuals)
class DSCBlock(nn.Module):
    """
    Depthwise Separable Convolution with Residual Connection.
    Standard in modern efficient networks (MobileNet, EfficientNet).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        
        # 1. Depthwise: Spatial convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 2. Pointwise: Channel mixing (1x1 conv)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True) # Swish activation

        # Skip connection logic
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        out += self.shortcut(x) # Residual Addition
        return self.act(out)

class CustomCNN(nn.Module):
    def __init__(self, num_classes, use_checkpointing=False):
        super(CustomCNN, self).__init__()
        self.use_checkpointing = use_checkpointing
        
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stacking optimized blocks (increasing channels, reducing size)
        # Input: 224x224
        self.layer1 = DSCBlock(32, 64, stride=2)   # -> 112x112
        self.layer2 = DSCBlock(64, 128, stride=2)  # -> 56x56
        self.layer3 = DSCBlock(128, 256, stride=2) # -> 28x28
        self.layer4 = DSCBlock(256, 512, stride=2) # -> 14x14
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        if self.use_checkpointing and x.requires_grad:
            x = checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
        x = torch.flatten(self.global_pool(x), 1)
        x = self.classifier(x)
        return x


#  CUSTOM TRANSFORMER (Pre-Norm + GELU)
class CustomTransformer(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16, 
                 embed_dim=256, num_heads=4, num_layers=6, dropout=0.1, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Patch calculations
        self.num_patches = (img_size // patch_size) ** 2
        
        # 1. Efficient Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2) # (B, C, N)
        )
        
        # 2. Learnable Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 3. Encoder with Pre-Norm (norm_first=True) and GELU
        # Pre-Norm is significantly more stable for deeper transformers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            activation='gelu', 
            batch_first=True,
            norm_first=True  # Critical optimization
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        # Good initialization is crucial for Transformers
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.patch_embed(x).transpose(1, 2) # (B, N, C)
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        if self.use_checkpointing and x.requires_grad:
            x = checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)
            
        x = self.norm(x)
        cls_out = x[:, 0] # Take only CLS token
        return self.head(cls_out)


#  VGG & TIMM MODELS
def build_optimized_vgg(model_name, num_classes, pretrained=True):
    if model_name == "vgg16":
        model = models.vgg16(weights='DEFAULT' if pretrained else None)
    elif model_name == "vgg19":
        model = models.vgg19(weights='DEFAULT' if pretrained else None)
    
    # Lightweight Head Optimization (for low VRAM)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model

def get_model_architecture(model_name, num_classes, pretrained=True, freeze_backbone=False, use_checkpointing=True):
    print(f"Initializing {model_name} (Checkpointing: {use_checkpointing})...")
    
    if model_name == "custom_cnn":
        return CustomCNN(num_classes, use_checkpointing=use_checkpointing)
    elif model_name == "custom_transformer":
        return CustomTransformer(num_classes=num_classes, use_checkpointing=use_checkpointing)
    elif "vgg" in model_name:
        return build_optimized_vgg(model_name, num_classes, pretrained)

    timm_map = {
        "mobilenet": "mobilenetv3_large_100",
        "efficientnetv2": "tf_efficientnetv2_s",
        "convnext": "convnext_tiny",
        "resnet50": "resnet50",
        "deit": "deit_tiny_patch16_224.fb_in1k",
        "deit_small": "deit_small_patch16_224.fb_in1k"
    }
    
    timm_name = timm_map.get(model_name, model_name)
    try:
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        if use_checkpointing:
            try:
                model.set_grad_checkpointing(enable=True)
            except: pass
        
        if freeze_backbone and pretrained:
            for p in model.parameters(): p.requires_grad = False
            for p in model.get_classifier().parameters(): p.requires_grad = True
                
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None