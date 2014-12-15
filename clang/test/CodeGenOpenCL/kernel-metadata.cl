// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void normal_function() {
}

__kernel void kernel_function() {
}

// CHECK: !opencl.kernels = !{!0}
// CHECK: !0 = !{void ()* @kernel_function, !1, !2, !3, !4, !5}
// CHECK: !1 = !{!"kernel_arg_addr_space"}
// CHECK: !2 = !{!"kernel_arg_access_qual"}
// CHECK: !3 = !{!"kernel_arg_type"}
// CHECK: !4 = !{!"kernel_arg_base_type"}
// CHECK: !5 = !{!"kernel_arg_type_qual"}
