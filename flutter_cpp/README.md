# SAM ONNX Integration for Flutter (C++ FFI)

This directory contains everything needed to run SAM segmentation natively in Flutter using ONNX Runtime and C++ FFI.

## 📁 Structure

```
flutter_cpp/
├── sam_inference.h      # C header (API definition)
├── sam_inference.cpp    # C++ implementation (ONNX Runtime)
├── sam_ffi.dart         # Dart FFI bindings
├── CMakeLists.txt       # Build configuration
└── README.md            # This file

onnx_models/
├── sam_encoder.onnx     # Image Encoder (~375 MB)
└── sam_decoder.onnx     # Mask Decoder (~16 MB)
```

## 🚀 Quick Start

### 1. Export ONNX Models (Already Done)
```bash
python export_onnx.py
```

### 2. Download ONNX Runtime

**Android:**
```bash
# Download from: https://github.com/microsoft/onnxruntime/releases
# Choose: onnxruntime-android-X.X.X.aar
```

**iOS:**
```bash
pod 'onnxruntime-objc', '~> 1.16.0'
```

**Windows/Linux/macOS:**
Download prebuilt from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

### 3. Build C++ Library

**Android (via NDK):**
```bash
mkdir build-android && cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DONNXRUNTIME_ROOT=/path/to/onnxruntime-android
make
```

**iOS (via Xcode):**
Add `sam_inference.cpp` and `sam_inference.h` to your Xcode project.

**Windows:**
```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=C:/path/to/onnxruntime
cmake --build . --config Release
```

### 4. Flutter Integration

1. Copy `sam_ffi.dart` to your Flutter project's `lib/` folder
2. Copy compiled library to:
   - Android: `android/app/src/main/jniLibs/<abi>/libsam_inference.so`
   - iOS: Add to Xcode project
   - Windows: Place `sam_inference.dll` next to executable
3. Copy ONNX models to Flutter assets or app documents directory

## 📝 Usage in Flutter

```dart
import 'sam_ffi.dart';

class FootSegmentation {
  final SamInference _sam = SamInference();
  
  Future<void> init() async {
    // Get model paths (from assets or documents)
    final encoderPath = await getModelPath('sam_encoder.onnx');
    final decoderPath = await getModelPath('sam_decoder.onnx');
    
    await _sam.initialize(encoderPath, decoderPath);
  }
  
  Future<Uint8List> segmentFoot(Uint8List imageRgb, int w, int h) async {
    // Smart points on foot (from coarse detection or user tap)
    final pointsX = [w * 0.5, w * 0.4, w * 0.6];
    final pointsY = [h * 0.6, h * 0.5, h * 0.7];
    final labels = [1, 1, 1]; // All foreground
    
    final result = await _sam.segment(
      imageRgb, w, h,
      pointsX, pointsY, labels,
    );
    
    print('Segmentation IoU: ${result.iouScore}');
    return result.mask;
  }
  
  void dispose() => _sam.dispose();
}
```

## ⚡ Performance Tips

1. **Encoder is heavy** (~500ms on mobile) - cache embeddings per image
2. **Decoder is light** (~50ms) - can run multiple times with different prompts
3. **Use fp16** on GPU-enabled devices for 2x speedup
4. **Quantize models** for smaller size:
   ```bash
   python -m onnxruntime.quantization.preprocess \
       --input sam_encoder.onnx \
       --output sam_encoder_prep.onnx
   
   python -m onnxruntime.quantization.quantize \
       --input sam_encoder_prep.onnx \
       --output sam_encoder_int8.onnx \
       --quantize_mode dynamic
   ```

## 🔄 Algorithm Match (Python ↔ C++)

The C++ implementation matches the Python pipeline exactly:

| Step | Python | C++ |
|------|--------|-----|
| Resize | `cv2.resize()` longest=1024 | Bilinear, longest=1024 |
| Normalize | `(x/255 - mean) / std` | Same formula |
| Encoder | `self.sam.image_encoder(x)` | `sam_encode_image()` |
| Points | `SamPredictor.predict()` | `sam_decode_mask()` |
| Output | `masks[best_idx]` | `result.masks[best_mask_idx]` |
| Postprocess | `cv2.resize()` + threshold | Bilinear + threshold |

**Normalization constants (ImageNet):**
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

## 🐛 Troubleshooting

### "Model not found"
Ensure ONNX files are accessible at runtime. On Android, copy from assets to app files directory first.

### "ONNX Runtime error"
Check ONNX Runtime version compatibility. Recommended: `1.16.x`

### "Wrong mask output"
Verify preprocessing matches Python exactly (resize, normalize, padding).

## 📊 Model Sizes

| Model | Size | Notes |
|-------|------|-------|
| `sam_encoder.onnx` | ~375 MB | Can quantize to ~95 MB |
| `sam_decoder.onnx` | ~16 MB | Keep FP32 for accuracy |

## 🔗 References

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Flutter FFI](https://dart.dev/guides/libraries/c-interop)
