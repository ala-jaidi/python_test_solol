"""
ONNX Export Script for SAM (Segment Anything Model)
Exports Image Encoder and Mask Decoder separately for Flutter C++ FFI integration.

Usage:
    python export_onnx.py

Output:
    - onnx_models/sam_encoder.onnx (Image Encoder - run once per image)
    - onnx_models/sam_decoder.onnx (Mask Decoder - run per prompt)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_mobile/sam_vit_b_mobile.pth"
OUTPUT_DIR = "onnx_models"
IMAGE_SIZE = 1024  # SAM input size

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE ENCODER WRAPPER
# ============================================================
class SAMImageEncoder(nn.Module):
    """Wrapper for SAM Image Encoder export"""
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        
    def forward(self, x):
        """
        Input: x - preprocessed image tensor [1, 3, 1024, 1024]
        Output: image_embeddings [1, 256, 64, 64]
        """
        return self.image_encoder(x)


# ============================================================
# MASK DECODER WRAPPER
# ============================================================
class SAMMaskDecoder(nn.Module):
    """Wrapper for SAM Mask Decoder export (with prompt encoder)"""
    def __init__(self, sam_model):
        super().__init__()
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        
        # Pre-compute no-mask embedding
        self.register_buffer(
            "no_mask_embed",
            sam_model.prompt_encoder.no_mask_embed.weight.reshape(1, 1, -1)
        )
        
    def forward(self, image_embeddings, point_coords, point_labels):
        """
        Inputs:
            - image_embeddings: [1, 256, 64, 64] from encoder
            - point_coords: [1, N, 2] point coordinates (x, y) in 1024x1024 space
            - point_labels: [1, N] labels (1=foreground, 0=background)
        
        Outputs:
            - masks: [1, 4, 256, 256] predicted masks (4 candidates)
            - iou_predictions: [1, 4] IoU scores for each mask
        """
        # Encode points
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )
        
        # Decode masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        
        return low_res_masks, iou_predictions


# ============================================================
# SIMPLIFIED DECODER (Points only, no box/mask prompts)
# ============================================================
class SAMMaskDecoderSimple(nn.Module):
    """
    Simplified Mask Decoder for point-only prompts.
    More efficient for Flutter/C++ integration.
    """
    def __init__(self, sam_model):
        super().__init__()
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        # Cache dense PE
        self.register_buffer("image_pe", sam_model.prompt_encoder.get_dense_pe())
        
        # Point embeddings
        self.point_embeddings = sam_model.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam_model.prompt_encoder.not_a_point_embed
        
    def _embed_points(self, point_coords, point_labels):
        """Embed point prompts"""
        # Normalize to [0, 1]
        point_coords = point_coords / IMAGE_SIZE
        
        # Get positional encoding
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        
        # Add label embeddings
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        
        return point_embedding
        
    def forward(self, image_embeddings, point_coords, point_labels):
        """
        Simplified forward pass for point prompts only.
        """
        # Encode points using prompt encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )
        
        # Decode
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        
        return low_res_masks, iou_predictions


# ============================================================
# EXPORT FUNCTIONS
# ============================================================
def export_encoder(sam_model, output_path):
    """Export Image Encoder to ONNX"""
    print("\n📦 Exporting Image Encoder...")
    
    encoder = SAMImageEncoder(sam_model)
    encoder.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # Export
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"}
        }
    )
    
    print(f"✅ Encoder exported: {output_path}")
    print(f"   Input: image [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"   Output: image_embeddings [1, 256, 64, 64]")
    
    return output_path


def export_decoder(sam_model, output_path):
    """Export Mask Decoder to ONNX"""
    print("\n📦 Exporting Mask Decoder...")
    
    decoder = SAMMaskDecoderSimple(sam_model)
    decoder.eval()
    
    # Dummy inputs
    dummy_embeddings = torch.randn(1, 256, 64, 64)
    dummy_points = torch.randint(0, IMAGE_SIZE, (1, 5, 2)).float()  # 5 points
    dummy_labels = torch.ones(1, 5).long()  # All foreground
    
    # Export
    torch.onnx.export(
        decoder,
        (dummy_embeddings, dummy_points, dummy_labels),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "image_embeddings": {0: "batch_size"},
            "point_coords": {0: "batch_size", 1: "num_points"},
            "point_labels": {0: "batch_size", 1: "num_points"},
            "masks": {0: "batch_size"},
            "iou_predictions": {0: "batch_size"}
        }
    )
    
    print(f"✅ Decoder exported: {output_path}")
    print(f"   Inputs:")
    print(f"     - image_embeddings [1, 256, 64, 64]")
    print(f"     - point_coords [1, N, 2]")
    print(f"     - point_labels [1, N]")
    print(f"   Outputs:")
    print(f"     - masks [1, 4, 256, 256]")
    print(f"     - iou_predictions [1, 4]")
    
    return output_path


def verify_onnx(onnx_path):
    """Verify ONNX model"""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"✅ ONNX model verified: {onnx_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SAM ONNX Export for Flutter C++ FFI")
    print("=" * 60)
    
    # Load model
    print(f"\n🔄 Loading SAM model ({MODEL_TYPE})...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.eval()
    print("✅ Model loaded")
    
    # Export Encoder
    encoder_path = os.path.join(OUTPUT_DIR, "sam_encoder.onnx")
    export_encoder(sam, encoder_path)
    
    # Export Decoder
    decoder_path = os.path.join(OUTPUT_DIR, "sam_decoder.onnx")
    export_decoder(sam, decoder_path)
    
    # Verify
    print("\n🔍 Verifying ONNX models...")
    try:
        import onnx
        verify_onnx(encoder_path)
        verify_onnx(decoder_path)
    except ImportError:
        print("⚠️ onnx package not installed, skipping verification")
        print("   Install with: pip install onnx==1.12.0")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles created in '{OUTPUT_DIR}/':")
    print(f"  1. sam_encoder.onnx - Image Encoder (~375 MB)")
    print(f"  2. sam_decoder.onnx - Mask Decoder (~16 MB)")
    
    print("\n📱 Flutter C++ Integration:")
    print("  1. Use ONNX Runtime C++ API")
    print("  2. Preprocess: Resize to 1024x1024, normalize [0,1], RGB")
    print("  3. Run encoder ONCE per image")
    print("  4. Run decoder for each point prompt")
    print("  5. Postprocess: Resize mask to original size, threshold > 0")
    
    print("\n📝 Preprocessing (must match Python):")
    print("  - Resize longest side to 1024, pad to square")
    print("  - Normalize: (pixel / 255.0 - mean) / std")
    print("  - Mean: [0.485, 0.456, 0.406]")
    print("  - Std:  [0.229, 0.224, 0.225]")


if __name__ == "__main__":
    main()
