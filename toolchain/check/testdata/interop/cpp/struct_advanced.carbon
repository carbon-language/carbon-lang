// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/struct_advanced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/struct_advanced.carbon

// ============================================================================
// Advanced C++ Struct Interoperability (Issue #5533)
// Based on carbon-interop Example 10: Enhanced C++ Struct Interoperability
// ============================================================================

// --- struct_advanced.h

// Simple struct with basic types
struct Point2D {
    double x, y;
};

// Nested struct composition
struct Rectangle {
    Point2D top_left;
    Point2D bottom_right;
    int id;
};

// Complex struct with various types
struct ImageData {
    unsigned int width, height;
    unsigned char channels;
    double* pixel_data;
    const char* metadata;
};

// C++ functions with struct parameters (Issue #5533)
auto process_point_by_value(Point2D point) -> double;
auto process_point_by_ref(const Point2D& point) -> double;
auto process_point_by_ptr(Point2D* point) -> void;

auto calculate_area(const Rectangle& rect) -> double;
auto resize_rectangle(Rectangle* rect, double scale) -> void;

auto get_image_info(const ImageData& img) -> int;
auto process_image_data(ImageData* img) -> bool;

// --- import_struct_advanced.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_advanced.h";

fn TestAdvancedStructInterop() {
  // Test simple struct by-value passing
  var point: auto = {.x = 10.5, .y = 20.7};
  var distance: f64 = Cpp.process_point_by_value(point);
  
  // Test struct by-reference passing (const reference)
  var distance_ref: f64 = Cpp.process_point_by_ref(point);
  
  // Test struct by-pointer passing
  Cpp.process_point_by_ptr(&point);
  
  // Test nested struct composition
  var rect: auto = {
    .top_left = {.x = 0.0, .y = 0.0},
    .bottom_right = {.x = 100.0, .y = 50.0},
    .id = 12345
  };
  
  var area: f64 = Cpp.calculate_area(rect);
  Cpp.resize_rectangle(&rect, 1.5);
  
  // Test complex struct with pointers and arrays
  var metadata: String = "RGB Image Data";
  var pixels: Array(f64) = Array(f64)();
  // Simulate pixel data
  var i: i32 = 0;
  while (i < 1000) {
    pixels.Push(Cast(f64, i) * 0.001);
    i = i + 1;
  }
  
  var image: auto = {
    .width = 640,
    .height = 480,
    .channels = 3,
    .pixel_data = pixels.Begin(),
    .metadata = metadata.c_str()
  };
  
  var info: i32 = Cpp.get_image_info(image);
  var success: Bool = Cpp.process_image_data(&image);
}

// --- struct_advanced.cpp

#include "struct_advanced.h"
#include <cmath>
#include <cstring>
#include <iostream>

auto process_point_by_value(Point2D point) -> double {
    return std::sqrt(point.x * point.x + point.y * point.y);
}

auto process_point_by_ref(const Point2D& point) -> double {
    return point.x + point.y;
}

auto process_point_by_ptr(Point2D* point) -> void {
    if (point) {
        point->x *= 2.0;
        point->y *= 2.0;
    }
}

auto calculate_area(const Rectangle& rect) -> double {
    double width = rect.bottom_right.x - rect.top_left.x;
    double height = rect.bottom_right.y - rect.top_left.y;
    return width * height;
}

auto resize_rectangle(Rectangle* rect, double scale) -> void {
    if (rect) {
        rect->bottom_right.x = rect->top_left.x + 
            (rect->bottom_right.x - rect->top_left.x) * scale;
        rect->bottom_right.y = rect->top_left.y + 
            (rect->bottom_right.y - rect->top_left.y) * scale;
    }
}

auto get_image_info(const ImageData& img) -> int {
    return static_cast<int>(img.width * img.height * img.channels);
}

auto process_image_data(ImageData* img) -> bool {
    if (!img || !img->pixel_data) {
        return false;
    }
    
    // Simulate image processing
    std::cout << "Processing " << img->width << "x" << img->height 
              << " image with " << static_cast<int>(img->channels) << " channels\n";
    
    if (img->metadata) {
        std::cout << "Metadata: " << img->metadata << "\n";
    }
    
    return true;
}

// CHECK:STDOUT: --- struct_advanced.h
// CHECK:STDOUT: 
// CHECK:STDOUT: // Simple struct with basic types
// CHECK:STDOUT: struct Point2D {
// CHECK:STDOUT:     double x, y;
// CHECK:STDOUT: };
// CHECK:STDOUT: 
// CHECK:STDOUT: // Nested struct composition
// CHECK:STDOUT: struct Rectangle {
// CHECK:STDOUT:     Point2D top_left;
// CHECK:STDOUT:     Point2D bottom_right;
// CHECK:STDOUT:     int id;
// CHECK:STDOUT: };
// CHECK:STDOUT: 
// CHECK:STDOUT: // Complex struct with various types
// CHECK:STDOUT: struct ImageData {
// CHECK:STDOUT:     unsigned int width, height;
// CHECK:STDOUT:     unsigned char channels;
// CHECK:STDOUT:     double* pixel_data;
// CHECK:STDOUT:     const char* metadata;
// CHECK:STDOUT: };
// CHECK:STDOUT: 
// CHECK:STDOUT: // C++ functions with struct parameters (Issue #5533)
// CHECK:STDOUT: auto process_point_by_value(Point2D point) -> double;
// CHECK:STDOUT: auto process_point_by_ref(const Point2D& point) -> double;
// CHECK:STDOUT: auto process_point_by_ptr(Point2D* point) -> void;
// CHECK:STDOUT: 
// CHECK:STDOUT: auto calculate_area(const Rectangle& rect) -> double;
// CHECK:STDOUT: auto resize_rectangle(Rectangle* rect, double scale) -> void;
// CHECK:STDOUT: 
// CHECK:STDOUT: auto get_image_info(const ImageData& img) -> int;
// CHECK:STDOUT: auto process_image_data(ImageData* img) -> bool;
// CHECK:STDOUT: 
// CHECK:STDOUT: --- import_struct_advanced.carbon
// CHECK:STDOUT: 
// CHECK:STDOUT: library "[[@TEST_NAME]]";
// CHECK:STDOUT: 
// CHECK:STDOUT: import Cpp library "struct_advanced.h";
// CHECK:STDOUT: 
// CHECK:STDOUT: fn TestAdvancedStructInterop() {
// CHECK:STDOUT:   // Test simple struct by-value passing
// CHECK:STDOUT:   var point: auto = {.x = 10.5, .y = 20.7};
// CHECK:STDOUT:   var distance: f64 = Cpp.process_point_by_value(point);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test struct by-reference passing (const reference)
// CHECK:STDOUT:   var distance_ref: f64 = Cpp.process_point_by_ref(point);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test struct by-pointer passing
// CHECK:STDOUT:   Cpp.process_point_by_ptr(&point);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test nested struct composition
// CHECK:STDOUT:   var rect: auto = {
// CHECK:STDOUT:     .top_left = {.x = 0.0, .y = 0.0},
// CHECK:STDOUT:     .bottom_right = {.x = 100.0, .y = 50.0},
// CHECK:STDOUT:     .id = 12345
// CHECK:STDOUT:   };
// CHECK:STDOUT:   
// CHECK:STDOUT:   var area: f64 = Cpp.calculate_area(rect);
// CHECK:STDOUT:   Cpp.resize_rectangle(&rect, 1.5);
// CHECK:STDOUT:   
// CHECK:STDOUT:   // Test complex struct with pointers and arrays
// CHECK:STDOUT:   var metadata: String = "RGB Image Data";
// CHECK:STDOUT:   var pixels: Array(f64) = Array(f64)();
// CHECK:STDOUT:   // Simulate pixel data
// CHECK:STDOUT:   var i: i32 = 0;
// CHECK:STDOUT:   while (i < 1000) {
// CHECK:STDOUT:     pixels.Push(Cast(f64, i) * 0.001);
// CHECK:STDOUT:     i = i + 1;
// CHECK:STDOUT:   }
// CHECK:STDOUT:   
// CHECK:STDOUT:   var image: auto = {
// CHECK:STDOUT:     .width = 640,
// CHECK:STDOUT:     .height = 480,
// CHECK:STDOUT:     .channels = 3,
// CHECK:STDOUT:     .pixel_data = pixels.Begin(),
// CHECK:STDOUT:     .metadata = metadata.c_str()
// CHECK:STDOUT:   };
// CHECK:STDOUT:   
// CHECK:STDOUT:   var info: i32 = Cpp.get_image_info(image);
// CHECK:STDOUT:   var success: Bool = Cpp.process_image_data(&image);
// CHECK:STDOUT: }