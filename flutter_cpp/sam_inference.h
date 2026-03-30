/**
 * SAM ONNX Inference for Flutter C++ FFI
 * 
 * This header defines the C interface for SAM model inference.
 * Use with ONNX Runtime C++ API.
 * 
 * Build: Compile as shared library (.so/.dll) and use dart:ffi
 */

#ifndef SAM_INFERENCE_H
#define SAM_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================
// CONSTANTS (Must match Python preprocessing)
// ============================================================
#define SAM_IMAGE_SIZE 1024
#define SAM_EMBEDDING_DIM 256
#define SAM_EMBEDDING_SIZE 64
#define SAM_MASK_SIZE 256
#define SAM_NUM_MASKS 4

// Normalization constants (ImageNet)
static const float SAM_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float SAM_STD[3] = {0.229f, 0.224f, 0.225f};

// ============================================================
// DATA STRUCTURES
// ============================================================

typedef struct {
    float* data;           // [1, 256, 64, 64] = 1,048,576 floats
    int batch_size;
    int channels;
    int height;
    int width;
} SamEmbedding;

typedef struct {
    float* coords;         // [N, 2] - (x, y) in 1024x1024 space
    int* labels;           // [N] - 1=foreground, 0=background
    int num_points;
} SamPointPrompt;

typedef struct {
    float* masks;          // [4, 256, 256] - 4 mask candidates
    float* iou_scores;     // [4] - confidence scores
    int best_mask_idx;     // Index of highest IoU mask
} SamMaskResult;

typedef struct {
    void* encoder_session;
    void* decoder_session;
    void* env;
    bool initialized;
} SamContext;

// ============================================================
// API FUNCTIONS (Export these via FFI)
// ============================================================

/**
 * Initialize SAM context with ONNX models
 * @param encoder_path Path to sam_encoder.onnx
 * @param decoder_path Path to sam_decoder.onnx
 * @return SamContext pointer (NULL on failure)
 */
SamContext* sam_init(const char* encoder_path, const char* decoder_path);

/**
 * Free SAM context
 */
void sam_free(SamContext* ctx);

/**
 * Preprocess image for SAM
 * @param rgb_data Raw RGB bytes [H, W, 3]
 * @param width Original image width
 * @param height Original image height
 * @param output Preallocated buffer [1, 3, 1024, 1024]
 * @param scale_x Output: x scale factor for coordinate mapping
 * @param scale_y Output: y scale factor for coordinate mapping
 */
void sam_preprocess_image(
    const uint8_t* rgb_data,
    int width,
    int height,
    float* output,
    float* scale_x,
    float* scale_y
);

/**
 * Run Image Encoder (HEAVY - call once per image)
 * @param ctx SAM context
 * @param preprocessed_image [1, 3, 1024, 1024] normalized tensor
 * @param embedding Output embedding (preallocated)
 * @return true on success
 */
bool sam_encode_image(
    SamContext* ctx,
    const float* preprocessed_image,
    SamEmbedding* embedding
);

/**
 * Run Mask Decoder (LIGHT - call per prompt)
 * @param ctx SAM context
 * @param embedding Image embedding from encoder
 * @param prompt Point prompt
 * @param result Output masks (preallocated)
 * @return true on success
 */
bool sam_decode_mask(
    SamContext* ctx,
    const SamEmbedding* embedding,
    const SamPointPrompt* prompt,
    SamMaskResult* result
);

/**
 * Postprocess mask to original image size
 * @param mask Low-res mask [256, 256]
 * @param output_width Target width
 * @param output_height Target height
 * @param output Preallocated buffer [output_height, output_width]
 * @param threshold Binarization threshold (default 0.0)
 */
void sam_postprocess_mask(
    const float* mask,
    int output_width,
    int output_height,
    uint8_t* output,
    float threshold
);

/**
 * Convert original image coordinates to SAM 1024x1024 space
 */
void sam_transform_coords(
    float orig_x, float orig_y,
    int orig_width, int orig_height,
    float* sam_x, float* sam_y
);

// ============================================================
// CONVENIENCE FUNCTION (All-in-one)
// ============================================================

/**
 * Full inference pipeline
 * @param ctx SAM context
 * @param rgb_data RGB image bytes
 * @param width Image width
 * @param height Image height
 * @param points_x X coordinates of prompt points
 * @param points_y Y coordinates of prompt points
 * @param labels Point labels (1=fg, 0=bg)
 * @param num_points Number of points
 * @param output_mask Preallocated mask buffer [height, width]
 * @return IoU score of best mask
 */
float sam_segment(
    SamContext* ctx,
    const uint8_t* rgb_data,
    int width,
    int height,
    const float* points_x,
    const float* points_y,
    const int* labels,
    int num_points,
    uint8_t* output_mask
);

#ifdef __cplusplus
}
#endif

#endif // SAM_INFERENCE_H
