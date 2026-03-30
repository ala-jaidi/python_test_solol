/**
 * ArUco L-Board Calibration for Flutter C++ FFI
 * 
 * Detects ArUco L-shaped board and calculates px/mm ratio.
 * Uses OpenCV ArUco module.
 */

#ifndef ARUCO_CALIBRATION_H
#define ARUCO_CALIBRATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================
// CONSTANTS (Must match Python)
// ============================================================
#define ARUCO_L_BOARD_SIZE_MM 60.0f
#define ARUCO_L_BOARD_SEPARATION_MM 12.0f
#define ARUCO_DICT_ID 10  // DICT_6X6_250

// Marker IDs in L-board
#define ARUCO_MARKER_CORNER 0
#define ARUCO_MARKER_X_AXIS 1
#define ARUCO_MARKER_Y_AXIS 2

// ============================================================
// DATA STRUCTURES
// ============================================================

typedef struct {
    float x;
    float y;
} Point2f;

typedef struct {
    Point2f corners[4];  // 4 corners of marker
    Point2f center;      // Center of marker
    int id;              // Marker ID
    bool detected;
} ArucoMarker;

typedef struct {
    float ratio_px_mm;           // Pixels per millimeter
    float distance_px;           // Distance in pixels between markers
    float known_distance_mm;     // Known distance in mm
    char used_pair[8];           // "0-1", "0-2", or "1-2"
    bool board_detected;
    ArucoMarker markers[3];      // Markers 0, 1, 2
    int num_markers_detected;
} ArucoCalibrationResult;

// ============================================================
// API FUNCTIONS
// ============================================================

/**
 * Detect ArUco L-board and calculate calibration ratio
 * 
 * @param rgb_data RGB image bytes [H, W, 3]
 * @param width Image width
 * @param height Image height
 * @param result Output calibration result
 * @return true if calibration successful (at least 2 markers detected)
 */
bool aruco_detect_l_board(
    const uint8_t* rgb_data,
    int width,
    int height,
    ArucoCalibrationResult* result
);

/**
 * Convert pixel measurement to millimeters
 */
float aruco_px_to_mm(float px, float ratio_px_mm);

/**
 * Convert millimeters to pixels
 */
float aruco_mm_to_px(float mm, float ratio_px_mm);

/**
 * Draw detected markers on image (for debug)
 * 
 * @param rgb_data RGB image bytes (will be modified)
 * @param width Image width
 * @param height Image height
 * @param result Calibration result with marker positions
 */
void aruco_draw_markers(
    uint8_t* rgb_data,
    int width,
    int height,
    const ArucoCalibrationResult* result
);

#ifdef __cplusplus
}
#endif

#endif // ARUCO_CALIBRATION_H
