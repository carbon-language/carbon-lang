// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

kernel  __attribute__((vec_type_hint(int))) __attribute__((reqd_work_group_size(1,2,4))) void kernel1(int a) {}

kernel __attribute__((vec_type_hint(uint4))) __attribute__((work_group_size_hint(8,16,32))) void kernel2(int a) {}

// CHECK: opencl.kernels = !{[[MDNODE0:![0-9]+]], [[MDNODE3:![0-9]+]]}

// CHECK: [[MDNODE0]] = metadata !{void (i32)* @kernel1, metadata [[MDNODE1:![0-9]+]], metadata [[MDNODE2:![0-9]+]]}
// CHECK: [[MDNODE1]] = metadata !{metadata !"vec_type_hint", i32 undef, i32 1}
// CHECK: [[MDNODE2]] = metadata !{metadata !"reqd_work_group_size", i32 1, i32 2, i32 4}
// CHECK: [[MDNODE3]] = metadata !{void (i32)* @kernel2, metadata [[MDNODE4:![0-9]+]], metadata [[MDNODE5:![0-9]+]]}
// CHECK: [[MDNODE4]] = metadata !{metadata !"vec_type_hint", <4 x i32> undef, i32 0}
// CHECK: [[MDNODE5]] = metadata !{metadata !"work_group_size_hint", i32 8, i32 16, i32 32}
