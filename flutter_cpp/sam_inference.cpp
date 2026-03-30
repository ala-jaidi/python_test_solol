/**
 * SAM ONNX Inference Implementation
 * 
 * Uses ONNX Runtime C++ API for inference.
 * Compile with: -lonnxruntime
 */

#include "sam_inference.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================
// INTERNAL STRUCTURES
// ============================================================

struct SamContextInternal {
    Ort::Env env;
    Ort::Session* encoder_session;
    Ort::Session* decoder_session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    
    SamContextInternal() : env(ORT_LOGGING_LEVEL_WARNING, "SAM") {
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
};

// ============================================================
// INITIALIZATION
// ============================================================

extern "C" SamContext* sam_init(const char* encoder_path, const char* decoder_path) {
    try {
        auto* internal = new SamContextInternal();
        
        // Load encoder
        internal->encoder_session = new Ort::Session(
            internal->env, 
            encoder_path, 
            internal->session_options
        );
        
        // Load decoder
        internal->decoder_session = new Ort::Session(
            internal->env, 
            decoder_path, 
            internal->session_options
        );
        
        auto* ctx = new SamContext();
        ctx->encoder_session = internal->encoder_session;
        ctx->decoder_session = internal->decoder_session;
        ctx->env = internal;
        ctx->initialized = true;
        
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void sam_free(SamContext* ctx) {
    if (ctx) {
        auto* internal = static_cast<SamContextInternal*>(ctx->env);
        delete static_cast<Ort::Session*>(ctx->encoder_session);
        delete static_cast<Ort::Session*>(ctx->decoder_session);
        delete internal;
        delete ctx;
    }
}

// ============================================================
// PREPROCESSING
// ============================================================

extern "C" void sam_preprocess_image(
    const uint8_t* rgb_data,
    int width,
    int height,
    float* output,
    float* scale_x,
    float* scale_y
) {
    // Calculate resize scale (longest side to 1024)
    float scale = static_cast<float>(SAM_IMAGE_SIZE) / std::max(width, height);
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);
    
    *scale_x = scale;
    *scale_y = scale;
    
    // Allocate resized buffer
    std::vector<float> resized(new_width * new_height * 3);
    
    // Bilinear resize
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float src_x = x / scale;
            float src_y = y / scale;
            
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);
            
            float wx = src_x - x0;
            float wy = src_y - y0;
            
            for (int c = 0; c < 3; c++) {
                float v00 = rgb_data[(y0 * width + x0) * 3 + c];
                float v01 = rgb_data[(y0 * width + x1) * 3 + c];
                float v10 = rgb_data[(y1 * width + x0) * 3 + c];
                float v11 = rgb_data[(y1 * width + x1) * 3 + c];
                
                float v = (1 - wx) * (1 - wy) * v00 +
                          wx * (1 - wy) * v01 +
                          (1 - wx) * wy * v10 +
                          wx * wy * v11;
                
                resized[(y * new_width + x) * 3 + c] = v;
            }
        }
    }
    
    // Normalize and copy to output (NCHW format with padding)
    std::memset(output, 0, SAM_IMAGE_SIZE * SAM_IMAGE_SIZE * 3 * sizeof(float));
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                float pixel = resized[(y * new_width + x) * 3 + c] / 255.0f;
                pixel = (pixel - SAM_MEAN[c]) / SAM_STD[c];
                output[c * SAM_IMAGE_SIZE * SAM_IMAGE_SIZE + y * SAM_IMAGE_SIZE + x] = pixel;
            }
        }
    }
}

// ============================================================
// ENCODER
// ============================================================

extern "C" bool sam_encode_image(
    SamContext* ctx,
    const float* preprocessed_image,
    SamEmbedding* embedding
) {
    if (!ctx || !ctx->initialized) return false;
    
    try {
        auto* session = static_cast<Ort::Session*>(ctx->encoder_session);
        auto* internal = static_cast<SamContextInternal*>(ctx->env);
        
        // Input tensor
        std::array<int64_t, 4> input_shape = {1, 3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(preprocessed_image),
            3 * SAM_IMAGE_SIZE * SAM_IMAGE_SIZE,
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"image"};
        const char* output_names[] = {"image_embeddings"};
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );
        
        // Copy output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = SAM_EMBEDDING_DIM * SAM_EMBEDDING_SIZE * SAM_EMBEDDING_SIZE;
        std::memcpy(embedding->data, output_data, output_size * sizeof(float));
        
        embedding->batch_size = 1;
        embedding->channels = SAM_EMBEDDING_DIM;
        embedding->height = SAM_EMBEDDING_SIZE;
        embedding->width = SAM_EMBEDDING_SIZE;
        
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================
// DECODER
// ============================================================

extern "C" bool sam_decode_mask(
    SamContext* ctx,
    const SamEmbedding* embedding,
    const SamPointPrompt* prompt,
    SamMaskResult* result
) {
    if (!ctx || !ctx->initialized) return false;
    
    try {
        auto* session = static_cast<Ort::Session*>(ctx->decoder_session);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Image embeddings tensor
        std::array<int64_t, 4> emb_shape = {1, SAM_EMBEDDING_DIM, SAM_EMBEDDING_SIZE, SAM_EMBEDDING_SIZE};
        Ort::Value emb_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(embedding->data),
            SAM_EMBEDDING_DIM * SAM_EMBEDDING_SIZE * SAM_EMBEDDING_SIZE,
            emb_shape.data(),
            emb_shape.size()
        );
        
        // Point coords tensor
        std::array<int64_t, 3> coords_shape = {1, prompt->num_points, 2};
        Ort::Value coords_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(prompt->coords),
            prompt->num_points * 2,
            coords_shape.data(),
            coords_shape.size()
        );
        
        // Point labels tensor (convert int to int64)
        std::vector<int64_t> labels_i64(prompt->num_points);
        for (int i = 0; i < prompt->num_points; i++) {
            labels_i64[i] = prompt->labels[i];
        }
        std::array<int64_t, 2> labels_shape = {1, prompt->num_points};
        Ort::Value labels_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            labels_i64.data(),
            prompt->num_points,
            labels_shape.data(),
            labels_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"image_embeddings", "point_coords", "point_labels"};
        const char* output_names[] = {"masks", "iou_predictions"};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(emb_tensor));
        input_tensors.push_back(std::move(coords_tensor));
        input_tensors.push_back(std::move(labels_tensor));
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            2
        );
        
        // Copy masks
        float* masks_data = output_tensors[0].GetTensorMutableData<float>();
        size_t masks_size = SAM_NUM_MASKS * SAM_MASK_SIZE * SAM_MASK_SIZE;
        std::memcpy(result->masks, masks_data, masks_size * sizeof(float));
        
        // Copy IoU scores and find best
        float* iou_data = output_tensors[1].GetTensorMutableData<float>();
        std::memcpy(result->iou_scores, iou_data, SAM_NUM_MASKS * sizeof(float));
        
        result->best_mask_idx = 0;
        float best_iou = iou_data[0];
        for (int i = 1; i < SAM_NUM_MASKS; i++) {
            if (iou_data[i] > best_iou) {
                best_iou = iou_data[i];
                result->best_mask_idx = i;
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================
// POSTPROCESSING
// ============================================================

extern "C" void sam_postprocess_mask(
    const float* mask,
    int output_width,
    int output_height,
    uint8_t* output,
    float threshold
) {
    // Bilinear resize from 256x256 to output size
    float scale_x = static_cast<float>(SAM_MASK_SIZE) / output_width;
    float scale_y = static_cast<float>(SAM_MASK_SIZE) / output_height;
    
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, SAM_MASK_SIZE - 1);
            int y1 = std::min(y0 + 1, SAM_MASK_SIZE - 1);
            
            float wx = src_x - x0;
            float wy = src_y - y0;
            
            float v = (1 - wx) * (1 - wy) * mask[y0 * SAM_MASK_SIZE + x0] +
                      wx * (1 - wy) * mask[y0 * SAM_MASK_SIZE + x1] +
                      (1 - wx) * wy * mask[y1 * SAM_MASK_SIZE + x0] +
                      wx * wy * mask[y1 * SAM_MASK_SIZE + x1];
            
            output[y * output_width + x] = (v > threshold) ? 255 : 0;
        }
    }
}

extern "C" void sam_transform_coords(
    float orig_x, float orig_y,
    int orig_width, int orig_height,
    float* sam_x, float* sam_y
) {
    float scale = static_cast<float>(SAM_IMAGE_SIZE) / std::max(orig_width, orig_height);
    *sam_x = orig_x * scale;
    *sam_y = orig_y * scale;
}

// ============================================================
// CONVENIENCE FUNCTION
// ============================================================

extern "C" float sam_segment(
    SamContext* ctx,
    const uint8_t* rgb_data,
    int width,
    int height,
    const float* points_x,
    const float* points_y,
    const int* labels,
    int num_points,
    uint8_t* output_mask
) {
    if (!ctx || !ctx->initialized || num_points == 0) return -1.0f;
    
    // Allocate buffers
    std::vector<float> preprocessed(3 * SAM_IMAGE_SIZE * SAM_IMAGE_SIZE);
    std::vector<float> embedding_data(SAM_EMBEDDING_DIM * SAM_EMBEDDING_SIZE * SAM_EMBEDDING_SIZE);
    std::vector<float> masks(SAM_NUM_MASKS * SAM_MASK_SIZE * SAM_MASK_SIZE);
    std::vector<float> iou_scores(SAM_NUM_MASKS);
    std::vector<float> coords(num_points * 2);
    std::vector<int> labels_copy(labels, labels + num_points);
    
    // Preprocess
    float scale_x, scale_y;
    sam_preprocess_image(rgb_data, width, height, preprocessed.data(), &scale_x, &scale_y);
    
    // Transform coordinates
    for (int i = 0; i < num_points; i++) {
        sam_transform_coords(points_x[i], points_y[i], width, height, &coords[i*2], &coords[i*2+1]);
    }
    
    // Encode
    SamEmbedding embedding = {embedding_data.data(), 1, SAM_EMBEDDING_DIM, SAM_EMBEDDING_SIZE, SAM_EMBEDDING_SIZE};
    if (!sam_encode_image(ctx, preprocessed.data(), &embedding)) {
        return -1.0f;
    }
    
    // Decode
    SamPointPrompt prompt = {coords.data(), labels_copy.data(), num_points};
    SamMaskResult result = {masks.data(), iou_scores.data(), 0};
    if (!sam_decode_mask(ctx, &embedding, &prompt, &result)) {
        return -1.0f;
    }
    
    // Postprocess best mask
    float* best_mask = masks.data() + result.best_mask_idx * SAM_MASK_SIZE * SAM_MASK_SIZE;
    sam_postprocess_mask(best_mask, width, height, output_mask, 0.0f);
    
    return iou_scores[result.best_mask_idx];
}
