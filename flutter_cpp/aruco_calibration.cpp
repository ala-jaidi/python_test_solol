/**
 * ArUco L-Board Calibration Implementation
 * 
 * Uses OpenCV ArUco module for marker detection.
 * Compile with: -lopencv_aruco -lopencv_core -lopencv_imgproc
 */

#include "aruco_calibration.h"
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <map>

// ============================================================
// HELPER FUNCTIONS
// ============================================================

static cv::Point2f compute_center(const std::vector<cv::Point2f>& corners) {
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center += corner;
    }
    return center / static_cast<float>(corners.size());
}

static float compute_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================
// ARUCO DETECTION
// ============================================================

extern "C" bool aruco_detect_l_board(
    const uint8_t* rgb_data,
    int width,
    int height,
    ArucoCalibrationResult* result
) {
    // Initialize result
    std::memset(result, 0, sizeof(ArucoCalibrationResult));
    result->board_detected = false;
    
    // Create OpenCV Mat from RGB data
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));
    
    // Convert RGB to BGR (OpenCV format)
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
    
    // Get ArUco dictionary (DICT_6X6_250)
    cv::Ptr<cv::aruco::Dictionary> dictionary = 
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    
    // Detector parameters
    cv::Ptr<cv::aruco::DetectorParameters> parameters = 
        cv::aruco::DetectorParameters::create();
    
    // Detect markers
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(bgr, dictionary, corners, ids, parameters);
    
    if (ids.empty()) {
        return false;
    }
    
    // Extract markers 0, 1, 2
    std::map<int, std::vector<cv::Point2f>> marker_positions;
    
    for (size_t i = 0; i < ids.size(); i++) {
        int marker_id = ids[i];
        if (marker_id >= 0 && marker_id <= 2) {
            marker_positions[marker_id] = corners[i];
            
            // Store in result
            ArucoMarker& marker = result->markers[marker_id];
            marker.id = marker_id;
            marker.detected = true;
            for (int j = 0; j < 4; j++) {
                marker.corners[j].x = corners[i][j].x;
                marker.corners[j].y = corners[i][j].y;
            }
            cv::Point2f center = compute_center(corners[i]);
            marker.center.x = center.x;
            marker.center.y = center.y;
            
            result->num_markers_detected++;
        }
    }
    
    // Need at least 2 markers
    if (marker_positions.size() < 2) {
        return false;
    }
    
    // Calculate known distance
    float known_distance_mm = ARUCO_L_BOARD_SIZE_MM + ARUCO_L_BOARD_SEPARATION_MM;
    float distance_px = 0;
    const char* used_pair = "";
    
    // Strategy: Try pairs in order of preference
    if (marker_positions.count(0) && marker_positions.count(1)) {
        // Case 1: 0-1 (main axis)
        cv::Point2f p0 = compute_center(marker_positions[0]);
        cv::Point2f p1 = compute_center(marker_positions[1]);
        distance_px = compute_distance(p0, p1);
        used_pair = "0-1";
    }
    else if (marker_positions.count(0) && marker_positions.count(2)) {
        // Case 2: 0-2 (secondary axis)
        cv::Point2f p0 = compute_center(marker_positions[0]);
        cv::Point2f p2 = compute_center(marker_positions[2]);
        distance_px = compute_distance(p0, p2);
        used_pair = "0-2";
    }
    else if (marker_positions.count(1) && marker_positions.count(2)) {
        // Case 3: 1-2 (diagonal)
        cv::Point2f p1 = compute_center(marker_positions[1]);
        cv::Point2f p2 = compute_center(marker_positions[2]);
        distance_px = compute_distance(p1, p2);
        known_distance_mm *= std::sqrt(2.0f);  // Diagonal distance
        used_pair = "1-2";
    }
    else {
        return false;
    }
    
    // Calculate ratio
    float ratio_px_mm = distance_px / known_distance_mm;
    
    // Fill result
    result->ratio_px_mm = ratio_px_mm;
    result->distance_px = distance_px;
    result->known_distance_mm = known_distance_mm;
    std::strncpy(result->used_pair, used_pair, sizeof(result->used_pair) - 1);
    result->board_detected = true;
    
    return true;
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

extern "C" float aruco_px_to_mm(float px, float ratio_px_mm) {
    if (ratio_px_mm <= 0) return 0;
    return px / ratio_px_mm;
}

extern "C" float aruco_mm_to_px(float mm, float ratio_px_mm) {
    return mm * ratio_px_mm;
}

extern "C" void aruco_draw_markers(
    uint8_t* rgb_data,
    int width,
    int height,
    const ArucoCalibrationResult* result
) {
    if (!result || !result->board_detected) return;
    
    cv::Mat image(height, width, CV_8UC3, rgb_data);
    
    // Draw each detected marker
    for (int i = 0; i < 3; i++) {
        const ArucoMarker& marker = result->markers[i];
        if (!marker.detected) continue;
        
        // Draw marker outline (yellow)
        std::vector<cv::Point> pts;
        for (int j = 0; j < 4; j++) {
            pts.push_back(cv::Point(
                static_cast<int>(marker.corners[j].x),
                static_cast<int>(marker.corners[j].y)
            ));
        }
        cv::polylines(image, pts, true, cv::Scalar(0, 255, 255), 3);
        
        // Draw center (red)
        cv::circle(image, 
            cv::Point(static_cast<int>(marker.center.x), static_cast<int>(marker.center.y)),
            5, cv::Scalar(0, 0, 255), -1);
        
        // Draw ID text
        char text[16];
        std::snprintf(text, sizeof(text), "ID:%d", marker.id);
        cv::putText(image, text,
            cv::Point(static_cast<int>(marker.center.x) + 10, static_cast<int>(marker.center.y)),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    }
    
    // Draw line between used pair
    int id1 = -1, id2 = -1;
    if (std::strcmp(result->used_pair, "0-1") == 0) { id1 = 0; id2 = 1; }
    else if (std::strcmp(result->used_pair, "0-2") == 0) { id1 = 0; id2 = 2; }
    else if (std::strcmp(result->used_pair, "1-2") == 0) { id1 = 1; id2 = 2; }
    
    if (id1 >= 0 && id2 >= 0 && result->markers[id1].detected && result->markers[id2].detected) {
        cv::line(image,
            cv::Point(static_cast<int>(result->markers[id1].center.x), 
                      static_cast<int>(result->markers[id1].center.y)),
            cv::Point(static_cast<int>(result->markers[id2].center.x), 
                      static_cast<int>(result->markers[id2].center.y)),
            cv::Scalar(0, 255, 0), 2);
    }
}
