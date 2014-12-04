// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void normal_function() {
}

__kernel void kernel_function() {
}

// CHECK: !opencl.kernels = !{!0}
// CHECK: !0 = metadata !{void ()* @kernel_function, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
// CHECK: !1 = metadata !{metadata !"kernel_arg_addr_space"}
// CHECK: !2 = metadata !{metadata !"kernel_arg_access_qual"}
// CHECK: !3 = metadata !{metadata !"kernel_arg_type"}
// CHECK: !4 = metadata !{metadata !"kernel_arg_base_type"}
// CHECK: !5 = metadata !{metadata !"kernel_arg_type_qual"}
