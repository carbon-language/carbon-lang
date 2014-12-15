// RUN: %clang_cc1 %s -cl-kernel-arg-info -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s -check-prefix ARGINFO
// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s -check-prefix NO-ARGINFO

kernel void foo(__global int * restrict X, const int Y, 
                volatile int anotherArg, __constant float * restrict Z) {
  *X = Y + anotherArg;
}

// CHECK:  !{!"kernel_arg_addr_space", i32 1, i32 0, i32 0, i32 2}
// CHECK:  !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none"}
// CHECK:  !{!"kernel_arg_type", !"int*", !"int", !"int", !"float*"}
// CHECK:  !{!"kernel_arg_base_type", !"int*", !"int", !"int", !"float*"}
// CHECK:  !{!"kernel_arg_type_qual", !"restrict", !"const", !"volatile", !"restrict const"}
// ARGINFO: !{!"kernel_arg_name", !"X", !"Y", !"anotherArg", !"Z"}
// NO-ARGINFO-NOT: !{!"kernel_arg_name", !"X", !"Y", !"anotherArg", !"Z"}

kernel void foo2(read_only image1d_t img1, image2d_t img2, write_only image2d_array_t img3) {
}
// CHECK:  !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
// CHECK:  !{!"kernel_arg_access_qual", !"read_only", !"read_only", !"write_only"}
// CHECK:  !{!"kernel_arg_type", !"image1d_t", !"image2d_t", !"image2d_array_t"}
// CHECK:  !{!"kernel_arg_base_type", !"image1d_t", !"image2d_t", !"image2d_array_t"}
// CHECK:  !{!"kernel_arg_type_qual", !"", !"", !""}
// ARGINFO: !{!"kernel_arg_name", !"img1", !"img2", !"img3"}
// NO-ARGINFO-NOT: !{!"kernel_arg_name", !"img1", !"img2", !"img3"}

kernel void foo3(__global half * X) {
}
// CHECK:  !{!"kernel_arg_addr_space", i32 1}
// CHECK:  !{!"kernel_arg_access_qual", !"none"}
// CHECK:  !{!"kernel_arg_type", !"half*"}
// CHECK:  !{!"kernel_arg_base_type", !"half*"}
// CHECK:  !{!"kernel_arg_type_qual", !""}
// ARGINFO: !{!"kernel_arg_name", !"X"}
// NO-ARGINFO-NOT: !{!"kernel_arg_name", !"X"}

typedef unsigned int myunsignedint;
kernel void foo4(__global unsigned int * X, __global myunsignedint * Y) {
}
// CHECK:  !{!"kernel_arg_addr_space", i32 1, i32 1}
// CHECK:  !{!"kernel_arg_access_qual", !"none", !"none"}
// CHECK:  !{!"kernel_arg_type", !"uint*", !"myunsignedint*"}
// CHECK:  !{!"kernel_arg_base_type", !"uint*", !"uint*"}
// CHECK:  !{!"kernel_arg_type_qual", !"", !""}
// ARGINFO: !{!"kernel_arg_name", !"X", !"Y"}
// NO-ARGINFO-NOT: !{!"kernel_arg_name", !"X", !"Y"}

typedef image1d_t myImage;
kernel void foo5(read_only myImage img1, write_only image1d_t img2) {
}
// CHECK:  !{!"kernel_arg_access_qual", !"read_only", !"write_only"}
// CHECK:  !{!"kernel_arg_type", !"myImage", !"image1d_t"}
// CHECK:  !{!"kernel_arg_base_type", !"image1d_t", !"image1d_t"}
// ARGINFO: !{!"kernel_arg_name", !"img1", !"img2"}
// NO-ARGINFO-NOT: !{!"kernel_arg_name", !"img1", !"img2"}
