/// SAM ONNX FFI Bindings for Flutter
/// 
/// Usage:
/// ```dart
/// final sam = SamInference();
/// await sam.initialize('assets/sam_encoder.onnx', 'assets/sam_decoder.onnx');
/// final mask = await sam.segment(imageBytes, width, height, points, labels);
/// sam.dispose();
/// ```

import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// ============================================================
// CONSTANTS
// ============================================================
const int SAM_IMAGE_SIZE = 1024;
const int SAM_EMBEDDING_DIM = 256;
const int SAM_EMBEDDING_SIZE = 64;
const int SAM_MASK_SIZE = 256;
const int SAM_NUM_MASKS = 4;

// ============================================================
// NATIVE STRUCT DEFINITIONS
// ============================================================

/// SamEmbedding struct
final class SamEmbedding extends Struct {
  external Pointer<Float> data;
  @Int32()
  external int batchSize;
  @Int32()
  external int channels;
  @Int32()
  external int height;
  @Int32()
  external int width;
}

/// SamPointPrompt struct
final class SamPointPrompt extends Struct {
  external Pointer<Float> coords;
  external Pointer<Int32> labels;
  @Int32()
  external int numPoints;
}

/// SamMaskResult struct
final class SamMaskResult extends Struct {
  external Pointer<Float> masks;
  external Pointer<Float> iouScores;
  @Int32()
  external int bestMaskIdx;
}

/// SamContext struct (opaque)
final class SamContext extends Opaque {}

// ============================================================
// NATIVE FUNCTION SIGNATURES
// ============================================================

typedef SamInitNative = Pointer<SamContext> Function(
  Pointer<Utf8> encoderPath,
  Pointer<Utf8> decoderPath,
);
typedef SamInitDart = Pointer<SamContext> Function(
  Pointer<Utf8> encoderPath,
  Pointer<Utf8> decoderPath,
);

typedef SamFreeNative = Void Function(Pointer<SamContext> ctx);
typedef SamFreeDart = void Function(Pointer<SamContext> ctx);

typedef SamPreprocessImageNative = Void Function(
  Pointer<Uint8> rgbData,
  Int32 width,
  Int32 height,
  Pointer<Float> output,
  Pointer<Float> scaleX,
  Pointer<Float> scaleY,
);
typedef SamPreprocessImageDart = void Function(
  Pointer<Uint8> rgbData,
  int width,
  int height,
  Pointer<Float> output,
  Pointer<Float> scaleX,
  Pointer<Float> scaleY,
);

typedef SamEncodeImageNative = Bool Function(
  Pointer<SamContext> ctx,
  Pointer<Float> preprocessedImage,
  Pointer<SamEmbedding> embedding,
);
typedef SamEncodeImageDart = bool Function(
  Pointer<SamContext> ctx,
  Pointer<Float> preprocessedImage,
  Pointer<SamEmbedding> embedding,
);

typedef SamDecodeMaskNative = Bool Function(
  Pointer<SamContext> ctx,
  Pointer<SamEmbedding> embedding,
  Pointer<SamPointPrompt> prompt,
  Pointer<SamMaskResult> result,
);
typedef SamDecodeMaskDart = bool Function(
  Pointer<SamContext> ctx,
  Pointer<SamEmbedding> embedding,
  Pointer<SamPointPrompt> prompt,
  Pointer<SamMaskResult> result,
);

typedef SamPostprocessMaskNative = Void Function(
  Pointer<Float> mask,
  Int32 outputWidth,
  Int32 outputHeight,
  Pointer<Uint8> output,
  Float threshold,
);
typedef SamPostprocessMaskDart = void Function(
  Pointer<Float> mask,
  int outputWidth,
  int outputHeight,
  Pointer<Uint8> output,
  double threshold,
);

typedef SamSegmentNative = Float Function(
  Pointer<SamContext> ctx,
  Pointer<Uint8> rgbData,
  Int32 width,
  Int32 height,
  Pointer<Float> pointsX,
  Pointer<Float> pointsY,
  Pointer<Int32> labels,
  Int32 numPoints,
  Pointer<Uint8> outputMask,
);
typedef SamSegmentDart = double Function(
  Pointer<SamContext> ctx,
  Pointer<Uint8> rgbData,
  int width,
  int height,
  Pointer<Float> pointsX,
  Pointer<Float> pointsY,
  Pointer<Int32> labels,
  int numPoints,
  Pointer<Uint8> outputMask,
);

// ============================================================
// SAM INFERENCE CLASS
// ============================================================

class SamInference {
  late DynamicLibrary _lib;
  Pointer<SamContext>? _ctx;
  
  // Cached native functions
  late SamInitDart _samInit;
  late SamFreeDart _samFree;
  late SamPreprocessImageDart _samPreprocessImage;
  late SamEncodeImageDart _samEncodeImage;
  late SamDecodeMaskDart _samDecodeMask;
  late SamPostprocessMaskDart _samPostprocessMask;
  late SamSegmentDart _samSegment;
  
  // Cached embedding for reuse
  Pointer<Float>? _cachedEmbedding;
  int _cachedImageHash = 0;
  
  bool get isInitialized => _ctx != null;
  
  /// Load the native library
  SamInference() {
    if (Platform.isAndroid) {
      _lib = DynamicLibrary.open('libsam_inference.so');
    } else if (Platform.isIOS) {
      _lib = DynamicLibrary.process();
    } else if (Platform.isWindows) {
      _lib = DynamicLibrary.open('sam_inference.dll');
    } else if (Platform.isLinux) {
      _lib = DynamicLibrary.open('libsam_inference.so');
    } else if (Platform.isMacOS) {
      _lib = DynamicLibrary.open('libsam_inference.dylib');
    } else {
      throw UnsupportedError('Platform not supported');
    }
    
    _bindFunctions();
  }
  
  void _bindFunctions() {
    _samInit = _lib.lookupFunction<SamInitNative, SamInitDart>('sam_init');
    _samFree = _lib.lookupFunction<SamFreeNative, SamFreeDart>('sam_free');
    _samPreprocessImage = _lib.lookupFunction<SamPreprocessImageNative, SamPreprocessImageDart>('sam_preprocess_image');
    _samEncodeImage = _lib.lookupFunction<SamEncodeImageNative, SamEncodeImageDart>('sam_encode_image');
    _samDecodeMask = _lib.lookupFunction<SamDecodeMaskNative, SamDecodeMaskDart>('sam_decode_mask');
    _samPostprocessMask = _lib.lookupFunction<SamPostprocessMaskNative, SamPostprocessMaskDart>('sam_postprocess_mask');
    _samSegment = _lib.lookupFunction<SamSegmentNative, SamSegmentDart>('sam_segment');
  }
  
  /// Initialize SAM with ONNX model paths
  Future<bool> initialize(String encoderPath, String decoderPath) async {
    final encoderPathPtr = encoderPath.toNativeUtf8();
    final decoderPathPtr = decoderPath.toNativeUtf8();
    
    try {
      _ctx = _samInit(encoderPathPtr, decoderPathPtr);
      return _ctx != null && _ctx != nullptr;
    } finally {
      calloc.free(encoderPathPtr);
      calloc.free(decoderPathPtr);
    }
  }
  
  /// Segment image with point prompts (all-in-one)
  /// 
  /// [rgbBytes] - RGB image data (H * W * 3)
  /// [width] - Image width
  /// [height] - Image height
  /// [pointsX] - X coordinates of prompt points
  /// [pointsY] - Y coordinates of prompt points
  /// [labels] - Point labels (1=foreground, 0=background)
  /// 
  /// Returns: Binary mask as Uint8List (H * W)
  Future<SegmentResult> segment(
    Uint8List rgbBytes,
    int width,
    int height,
    List<double> pointsX,
    List<double> pointsY,
    List<int> labels,
  ) async {
    if (_ctx == null) {
      throw StateError('SAM not initialized. Call initialize() first.');
    }
    
    if (pointsX.length != pointsY.length || pointsX.length != labels.length) {
      throw ArgumentError('Points and labels must have same length');
    }
    
    final numPoints = pointsX.length;
    
    // Allocate native memory
    final rgbPtr = calloc<Uint8>(rgbBytes.length);
    final pointsXPtr = calloc<Float>(numPoints);
    final pointsYPtr = calloc<Float>(numPoints);
    final labelsPtr = calloc<Int32>(numPoints);
    final maskPtr = calloc<Uint8>(width * height);
    
    try {
      // Copy data to native memory
      rgbPtr.asTypedList(rgbBytes.length).setAll(0, rgbBytes);
      pointsXPtr.asTypedList(numPoints).setAll(0, pointsX.map((e) => e).toList());
      pointsYPtr.asTypedList(numPoints).setAll(0, pointsY.map((e) => e).toList());
      labelsPtr.asTypedList(numPoints).setAll(0, labels);
      
      // Run inference
      final iou = _samSegment(
        _ctx!,
        rgbPtr,
        width,
        height,
        pointsXPtr,
        pointsYPtr,
        labelsPtr,
        numPoints,
        maskPtr,
      );
      
      if (iou < 0) {
        throw Exception('Segmentation failed');
      }
      
      // Copy result
      final mask = Uint8List.fromList(maskPtr.asTypedList(width * height));
      
      return SegmentResult(mask: mask, iouScore: iou);
    } finally {
      calloc.free(rgbPtr);
      calloc.free(pointsXPtr);
      calloc.free(pointsYPtr);
      calloc.free(labelsPtr);
      calloc.free(maskPtr);
    }
  }
  
  /// Dispose resources
  void dispose() {
    if (_ctx != null) {
      _samFree(_ctx!);
      _ctx = null;
    }
    if (_cachedEmbedding != null) {
      calloc.free(_cachedEmbedding!);
      _cachedEmbedding = null;
    }
  }
}

/// Result of segmentation
class SegmentResult {
  final Uint8List mask;
  final double iouScore;
  
  SegmentResult({required this.mask, required this.iouScore});
}

// ============================================================
// ARUCO CALIBRATION
// ============================================================

// ArUco constants
const double ARUCO_L_BOARD_SIZE_MM = 60.0;
const double ARUCO_L_BOARD_SEPARATION_MM = 12.0;

/// Point2f struct
final class Point2f extends Struct {
  @Float()
  external double x;
  @Float()
  external double y;
}

/// ArucoMarker struct
final class ArucoMarker extends Struct {
  @Array(4)
  external Array<Point2f> corners;
  external Point2f center;
  @Int32()
  external int id;
  @Bool()
  external bool detected;
}

/// ArucoCalibrationResult struct
final class ArucoCalibrationResult extends Struct {
  @Float()
  external double ratioPxMm;
  @Float()
  external double distancePx;
  @Float()
  external double knownDistanceMm;
  @Array(8)
  external Array<Uint8> usedPair;
  @Bool()
  external bool boardDetected;
  @Array(3)
  external Array<ArucoMarker> markers;
  @Int32()
  external int numMarkersDetected;
}

// ArUco native function signatures
typedef ArucoDetectNative = Bool Function(
  Pointer<Uint8> rgbData,
  Int32 width,
  Int32 height,
  Pointer<ArucoCalibrationResult> result,
);
typedef ArucoDetectDart = bool Function(
  Pointer<Uint8> rgbData,
  int width,
  int height,
  Pointer<ArucoCalibrationResult> result,
);

typedef ArucoPxToMmNative = Float Function(Float px, Float ratio);
typedef ArucoPxToMmDart = double Function(double px, double ratio);

/// ArUco calibration class
class ArucoCalibration {
  late DynamicLibrary _lib;
  late ArucoDetectDart _arucoDetect;
  late ArucoPxToMmDart _arucoPxToMm;
  
  ArucoCalibration(DynamicLibrary lib) {
    _lib = lib;
    _arucoDetect = _lib.lookupFunction<ArucoDetectNative, ArucoDetectDart>('aruco_detect_l_board');
    _arucoPxToMm = _lib.lookupFunction<ArucoPxToMmNative, ArucoPxToMmDart>('aruco_px_to_mm');
  }
  
  /// Detect ArUco L-board and get calibration ratio
  CalibrationResult? detectLBoard(Uint8List rgbBytes, int width, int height) {
    final rgbPtr = calloc<Uint8>(rgbBytes.length);
    final resultPtr = calloc<ArucoCalibrationResult>();
    
    try {
      rgbPtr.asTypedList(rgbBytes.length).setAll(0, rgbBytes);
      
      final success = _arucoDetect(rgbPtr, width, height, resultPtr);
      
      if (!success) return null;
      
      final result = resultPtr.ref;
      return CalibrationResult(
        ratioPxMm: result.ratioPxMm,
        distancePx: result.distancePx,
        knownDistanceMm: result.knownDistanceMm,
        usedPair: String.fromCharCodes(
          result.usedPair.asTypedList(8).takeWhile((c) => c != 0)
        ),
        numMarkersDetected: result.numMarkersDetected,
      );
    } finally {
      calloc.free(rgbPtr);
      calloc.free(resultPtr);
    }
  }
  
  /// Convert pixels to millimeters
  double pxToMm(double px, double ratioPxMm) {
    return _arucoPxToMm(px, ratioPxMm);
  }
  
  /// Convert millimeters to pixels
  double mmToPx(double mm, double ratioPxMm) {
    return mm * ratioPxMm;
  }
}

/// Calibration result
class CalibrationResult {
  final double ratioPxMm;
  final double distancePx;
  final double knownDistanceMm;
  final String usedPair;
  final int numMarkersDetected;
  
  CalibrationResult({
    required this.ratioPxMm,
    required this.distancePx,
    required this.knownDistanceMm,
    required this.usedPair,
    required this.numMarkersDetected,
  });
  
  @override
  String toString() => 'CalibrationResult(ratio: $ratioPxMm px/mm, pair: $usedPair)';
}

// ============================================================
// COMPLETE PODIATRY PIPELINE
// ============================================================

/// Complete podiatry measurement pipeline
class PodiatryPipeline {
  final SamInference _sam;
  late ArucoCalibration _aruco;
  
  PodiatryPipeline() : _sam = SamInference() {
    _aruco = ArucoCalibration(_sam._lib);
  }
  
  /// Initialize with ONNX model paths
  Future<bool> initialize(String encoderPath, String decoderPath) async {
    return await _sam.initialize(encoderPath, decoderPath);
  }
  
  /// Process side view (length measurement)
  Future<SideViewResult?> processSideView(
    Uint8List rgbBytes,
    int width,
    int height,
    String footSide, // "left" or "right"
  ) async {
    // 1. Detect ArUco for calibration
    final calibration = _aruco.detectLBoard(rgbBytes, width, height);
    if (calibration == null) {
      return null; // No calibration reference found
    }
    
    // 2. Segment foot (use center points as initial prompt)
    final pointsX = [width * 0.5];
    final pointsY = [height * 0.6];
    final labels = [1];
    
    final segResult = await _sam.segment(
      rgbBytes, width, height,
      pointsX, pointsY, labels,
    );
    
    // 3. Find heel and toe from mask
    final heelToe = _findHeelAndToe(segResult.mask, width, height, footSide);
    if (heelToe == null) return null;
    
    // 4. Calculate length
    final lengthPx = _distance(heelToe.heel, heelToe.toe);
    final lengthMm = _aruco.pxToMm(lengthPx, calibration.ratioPxMm);
    final lengthCm = lengthMm / 10.0;
    
    return SideViewResult(
      lengthCm: lengthCm,
      heelPoint: heelToe.heel,
      toePoint: heelToe.toe,
      calibration: calibration,
      mask: segResult.mask,
    );
  }
  
  /// Process top view (width measurement)
  Future<TopViewResult?> processTopView(
    Uint8List rgbBytes,
    int width,
    int height,
  ) async {
    // 1. Detect ArUco for calibration
    final calibration = _aruco.detectLBoard(rgbBytes, width, height);
    if (calibration == null) {
      return null;
    }
    
    // 2. Segment foot
    final pointsX = [width * 0.5];
    final pointsY = [height * 0.5];
    final labels = [1];
    
    final segResult = await _sam.segment(
      rgbBytes, width, height,
      pointsX, pointsY, labels,
    );
    
    // 3. Find max width from mask
    final widthPoints = _findMaxWidth(segResult.mask, width, height);
    if (widthPoints == null) return null;
    
    // 4. Calculate width
    final widthPx = _distance(widthPoints.left, widthPoints.right);
    final widthMm = _aruco.pxToMm(widthPx, calibration.ratioPxMm);
    final widthCm = widthMm / 10.0;
    
    return TopViewResult(
      widthCm: widthCm,
      leftPoint: widthPoints.left,
      rightPoint: widthPoints.right,
      calibration: calibration,
      mask: segResult.mask,
    );
  }
  
  // Helper: Find heel and toe points from mask
  _HeelToe? _findHeelAndToe(Uint8List mask, int w, int h, String footSide) {
    int minX = w, maxX = 0;
    int minXY = 0, maxXY = 0;
    
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (mask[y * w + x] > 0) {
          if (x < minX) { minX = x; minXY = y; }
          if (x > maxX) { maxX = x; maxXY = y; }
        }
      }
    }
    
    if (minX >= maxX) return null;
    
    // Left foot: toe at min X, heel at max X
    // Right foot: heel at min X, toe at max X
    if (footSide.toLowerCase() == "left") {
      return _HeelToe(
        heel: Point(maxX.toDouble(), maxXY.toDouble()),
        toe: Point(minX.toDouble(), minXY.toDouble()),
      );
    } else {
      return _HeelToe(
        heel: Point(minX.toDouble(), minXY.toDouble()),
        toe: Point(maxX.toDouble(), maxXY.toDouble()),
      );
    }
  }
  
  // Helper: Find max width points from mask
  _WidthPoints? _findMaxWidth(Uint8List mask, int w, int h) {
    double maxWidth = 0;
    int bestY = 0;
    int bestLeft = 0, bestRight = 0;
    
    for (int y = 0; y < h; y++) {
      int left = -1, right = -1;
      for (int x = 0; x < w; x++) {
        if (mask[y * w + x] > 0) {
          if (left < 0) left = x;
          right = x;
        }
      }
      if (left >= 0 && right > left) {
        double width = (right - left).toDouble();
        if (width > maxWidth) {
          maxWidth = width;
          bestY = y;
          bestLeft = left;
          bestRight = right;
        }
      }
    }
    
    if (maxWidth == 0) return null;
    
    return _WidthPoints(
      left: Point(bestLeft.toDouble(), bestY.toDouble()),
      right: Point(bestRight.toDouble(), bestY.toDouble()),
    );
  }
  
  double _distance(Point p1, Point p2) {
    final dx = p2.x - p1.x;
    final dy = p2.y - p1.y;
    return math.sqrt(dx * dx + dy * dy);
  }
  
  void dispose() => _sam.dispose();
}

// Helper classes
class Point {
  final double x, y;
  Point(this.x, this.y);
}

class _HeelToe {
  final Point heel, toe;
  _HeelToe({required this.heel, required this.toe});
}

class _WidthPoints {
  final Point left, right;
  _WidthPoints({required this.left, required this.right});
}

class SideViewResult {
  final double lengthCm;
  final Point heelPoint;
  final Point toePoint;
  final CalibrationResult calibration;
  final Uint8List mask;
  
  SideViewResult({
    required this.lengthCm,
    required this.heelPoint,
    required this.toePoint,
    required this.calibration,
    required this.mask,
  });
}

class TopViewResult {
  final double widthCm;
  final Point leftPoint;
  final Point rightPoint;
  final CalibrationResult calibration;
  final Uint8List mask;
  
  TopViewResult({
    required this.widthCm,
    required this.leftPoint,
    required this.rightPoint,
    required this.calibration,
    required this.mask,
  });
}

double _sqrt(double x) => math.sqrt(x);

// ============================================================
// USAGE EXAMPLE
// ============================================================
/*
void main() async {
  final sam = SamInference();
  
  // Initialize with model paths
  final success = await sam.initialize(
    '/path/to/sam_encoder.onnx',
    '/path/to/sam_decoder.onnx',
  );
  
  if (!success) {
    print('Failed to initialize SAM');
    return;
  }
  
  // Load image (RGB bytes)
  final imageBytes = ...; // Uint8List of RGB data
  final width = 640;
  final height = 480;
  
  // Define prompt points (click on foot)
  final pointsX = [320.0, 350.0]; // X coordinates
  final pointsY = [240.0, 280.0]; // Y coordinates
  final labels = [1, 1]; // All foreground points
  
  // Segment
  final result = await sam.segment(
    imageBytes,
    width,
    height,
    pointsX,
    pointsY,
    labels,
  );
  
  print('IoU Score: ${result.iouScore}');
  print('Mask size: ${result.mask.length}');
  
  // Use mask for further processing...
  
  // Cleanup
  sam.dispose();
}
*/
