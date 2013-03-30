// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -emit-llvm -O0 -o - | FileCheck %s

void device_function() {
}
// CHECK: define void @device_function()

__kernel void kernel_function() {
  device_function();
}
// CHECK: define void @kernel_function()
// CHECK: call void @device_function()
// CHECK: !{{[0-9]+}} = metadata !{void ()* @kernel_function, metadata !"kernel", i32 1}

