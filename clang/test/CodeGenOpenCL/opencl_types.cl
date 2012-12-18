// RUN: %clang_cc1 %s -emit-llvm -o - -O0 | FileCheck %s

void fnc1(image1d_t img) {}
// CHECK: @fnc1(%opencl.image1d_t*

void fnc1arr(image1d_array_t img) {}
// CHECK: @fnc1arr(%opencl.image1d_array_t*

void fnc1buff(image1d_buffer_t img) {}
// CHECK: @fnc1buff(%opencl.image1d_buffer_t*

void fnc2(image2d_t img) {}
// CHECK: @fnc2(%opencl.image2d_t*

void fnc2arr(image2d_array_t img) {}
// CHECK: @fnc2arr(%opencl.image2d_array_t*

void fnc3(image3d_t img) {}
// CHECK: @fnc3(%opencl.image3d_t*

kernel void foo(image1d_t img) {
}
