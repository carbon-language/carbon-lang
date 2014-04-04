// RUN: %clang_cc1 %s -cl-kernel-arg-info -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s

kernel void foo(__global int * restrict X, const int Y, 
                volatile int anotherArg, __constant float * restrict Z) {
  *X = Y + anotherArg;
}

// CHECK: metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 0, i32 0, i32 2}
// CHECK: metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
// CHECK: metadata !{metadata !"kernel_arg_type", metadata !"int*", metadata !"int", metadata !"int", metadata !"float*"}
// CHECK: metadata !{metadata !"kernel_arg_type_qual", metadata !"restrict", metadata !"const", metadata !"volatile", metadata !"restrict const"}
// CHECK: metadata !{metadata !"kernel_arg_name", metadata !"X", metadata !"Y", metadata !"anotherArg", metadata !"Z"}

kernel void foo2(read_only image1d_t img1, image2d_t img2, write_only image2d_array_t img3) {
}
// CHECK: metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1, i32 1}
// CHECK: metadata !{metadata !"kernel_arg_access_qual", metadata !"read_only", metadata !"read_only", metadata !"write_only"}
// CHECK: metadata !{metadata !"kernel_arg_type", metadata !"image1d_t", metadata !"image2d_t", metadata !"image2d_array_t"}
// CHECK: metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !""}
// CHECK: metadata !{metadata !"kernel_arg_name", metadata !"img1", metadata !"img2", metadata !"img3"}

kernel void foo3(__global half * X) {
}
// CHECK: metadata !{metadata !"kernel_arg_addr_space", i32 1}
// CHECK: metadata !{metadata !"kernel_arg_access_qual", metadata !"none"}
// CHECK: metadata !{metadata !"kernel_arg_type", metadata !"half*"}
// CHECK: metadata !{metadata !"kernel_arg_type_qual", metadata !""}
// CHECK: metadata !{metadata !"kernel_arg_name", metadata !"X"}
