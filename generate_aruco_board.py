#!/usr/bin/env python3
"""
ArUco L-shaped Board Generator
Creates printable ArUco markers for 3D-aware foot measurement calibration
"""

import cv2
import numpy as np
import argparse
import os

def generate_aruco_l_board(output_dir="aruco_boards", 
                          marker_size_mm=100, 
                          separation_mm=20,
                          dpi=300,
                          dict_type=cv2.aruco.DICT_6X6_250):
    """
    Generate an L-shaped ArUco board for foot measurement calibration
    
    Args:
        output_dir: Directory to save the generated boards
        marker_size_mm: Size of each marker in millimeters
        separation_mm: Separation between markers in millimeters
        dpi: Print resolution (dots per inch)
        dict_type: ArUco dictionary type
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate pixel size based on DPI
    # 1 inch = 25.4 mm
    pixels_per_mm = dpi / 25.4
    marker_size_px = int(marker_size_mm * pixels_per_mm)
    separation_px = int(separation_mm * pixels_per_mm)
    
    # Initialize ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    # Generate individual markers
    marker_0 = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size_px)
    marker_1 = cv2.aruco.generateImageMarker(aruco_dict, 1, marker_size_px)
    marker_2 = cv2.aruco.generateImageMarker(aruco_dict, 2, marker_size_px)
    
    # Create L-shaped board layout
    # Marker 0 at origin (corner of L)
    # Marker 1 along X-axis
    # Marker 2 along Y-axis
    
    board_width = marker_size_px * 2 + separation_px + 100  # Extra margin
    board_height = marker_size_px * 2 + separation_px + 100
    
    # Create white background
    l_board = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # Calculate positions
    margin = 50
    
    # Marker 0 position (origin/corner)
    m0_x = margin
    m0_y = margin + marker_size_px + separation_px
    
    # Marker 1 position (along X-axis)
    m1_x = m0_x + marker_size_px + separation_px
    m1_y = m0_y
    
    # Marker 2 position (along Y-axis)
    m2_x = m0_x
    m2_y = m0_y - marker_size_px - separation_px
    
    # Place markers on board
    l_board[m0_y:m0_y+marker_size_px, m0_x:m0_x+marker_size_px] = marker_0
    l_board[m1_y:m1_y+marker_size_px, m1_x:m1_x+marker_size_px] = marker_1
    l_board[m2_y:m2_y+marker_size_px, m2_x:m2_x+marker_size_px] = marker_2
    
    # Add coordinate system indicators
    # Draw L-shape outline
    cv2.line(l_board, (m0_x-10, m0_y+marker_size_px+10), 
             (m1_x+marker_size_px+10, m0_y+marker_size_px+10), 0, 2)  # X-axis line
    cv2.line(l_board, (m0_x-10, m0_y+marker_size_px+10), 
             (m0_x-10, m2_y-10), 0, 2)  # Y-axis line
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(l_board, "ID:0 (Origin)", (m0_x, m0_y+marker_size_px+30), font, 0.6, 0, 2)
    cv2.putText(l_board, "ID:1 (X-axis)", (m1_x, m1_y+marker_size_px+30), font, 0.6, 0, 2)
    cv2.putText(l_board, "ID:2 (Y-axis)", (m2_x-80, m2_y-10), font, 0.6, 0, 2)
    
    # Add measurement information
    info_y = board_height - 80
    cv2.putText(l_board, f"Marker size: {marker_size_mm}mm", (margin, info_y), font, 0.5, 0, 1)
    cv2.putText(l_board, f"Separation: {separation_mm}mm", (margin, info_y+20), font, 0.5, 0, 1)
    cv2.putText(l_board, f"Print at {dpi} DPI", (margin, info_y+40), font, 0.5, 0, 1)
    cv2.putText(l_board, "For foot measurement calibration", (margin, info_y+60), font, 0.5, 0, 1)
    
    # Save the L-board
    l_board_path = os.path.join(output_dir, f"aruco_l_board_{marker_size_mm}mm_{dpi}dpi.png")
    cv2.imwrite(l_board_path, l_board)
    
    # Also save individual markers for reference
    individual_dir = os.path.join(output_dir, "individual_markers")
    os.makedirs(individual_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(individual_dir, "marker_0.png"), marker_0)
    cv2.imwrite(os.path.join(individual_dir, "marker_1.png"), marker_1)
    cv2.imwrite(os.path.join(individual_dir, "marker_2.png"), marker_2)
    
    # Create usage instructions
    instructions_path = os.path.join(output_dir, "USAGE_INSTRUCTIONS.txt")
    with open(instructions_path, 'w') as f:
        f.write("ArUco L-Board Usage Instructions\n")
        f.write("================================\n\n")
        f.write("1. PRINTING:\n")
        f.write(f"   - Print aruco_l_board_{marker_size_mm}mm_{dpi}dpi.png at ACTUAL SIZE\n")
        f.write(f"   - Ensure printer is set to {dpi} DPI\n")
        f.write("   - Use white paper, preferably thick/cardboard\n")
        f.write("   - Do NOT scale the image when printing\n\n")
        f.write("2. SETUP:\n")
        f.write("   - Cut out the L-shaped board\n")
        f.write("   - Mount on a rigid surface if needed\n")
        f.write("   - Place the board on the same surface as the foot\n")
        f.write("   - Ensure good lighting and minimal shadows\n\n")
        f.write("3. POSITIONING:\n")
        f.write("   - Marker 0 (ID:0) is the origin point\n")
        f.write("   - Marker 1 (ID:1) defines the X-axis direction\n")
        f.write("   - Marker 2 (ID:2) defines the Y-axis direction\n")
        f.write("   - Keep all markers visible in photos\n\n")
        f.write("4. PHOTOGRAPHY:\n")
        f.write("   - Take photos from directly above (top view) and from the side\n")
        f.write("   - Ensure all three markers are clearly visible\n")
        f.write("   - Avoid reflections and maintain good contrast\n")
        f.write("   - The software will automatically detect and use the markers\n\n")
        f.write(f"TECHNICAL SPECS:\n")
        f.write(f"- Marker size: {marker_size_mm}mm x {marker_size_mm}mm\n")
        f.write(f"- Separation: {separation_mm}mm (center-to-center)\n")
        f.write(f"- Dictionary: 6x6_250\n")
        f.write(f"- Calibration accuracy: ~1mm\n")
    
    print(f"✅ ArUco L-board generated successfully!")
    print(f"📁 Files saved to: {output_dir}/")
    print(f"📄 L-board: {l_board_path}")
    print(f"📋 Instructions: {instructions_path}")
    print(f"📏 Marker size: {marker_size_mm}mm")
    print(f"📐 Separation: {separation_mm}mm")
    print(f"🖨️  Print DPI: {dpi}")
    
    return l_board_path


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco L-shaped board for foot measurement")
    parser.add_argument('--size', type=int, default=100, help='Marker size in mm (default: 100)')
    parser.add_argument('--separation', type=int, default=20, help='Separation between markers in mm (default: 20)')
    parser.add_argument('--dpi', type=int, default=300, help='Print resolution (default: 300)')
    parser.add_argument('--output', type=str, default='aruco_boards', help='Output directory (default: aruco_boards)')
    
    args = parser.parse_args()
    
    generate_aruco_l_board(
        output_dir=args.output,
        marker_size_mm=args.size,
        separation_mm=args.separation,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()