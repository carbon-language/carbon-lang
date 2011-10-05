// RUN: %clang_cc1 %s -triple ptx32-unknown-unknown -emit-llvm -o - | FileCheck %s

void device_function() {
}
// CHECK: define ptx_device void @device_function()

__kernel void kernel_function() {
}
// CHECK: define ptx_kernel void @kernel_function()

