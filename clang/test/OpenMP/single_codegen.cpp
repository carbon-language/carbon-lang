// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }

// CHECK:       define void [[FOO:@.+]]()

void foo() {}

// CHECK-LABEL: @main
int main() {
  // CHECK:       [[A_ADDR:%.+]] = alloca i8
  char a;

// CHECK:       [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  [[IS_SINGLE:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_SINGLE]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  call void @__kmpc_end_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]
#pragma omp single
  a = 2;
// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  [[IS_SINGLE:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_SINGLE]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  call void [[FOO]]()
// CHECK-NEXT:  call void @__kmpc_end_single([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]
#pragma omp single
  foo();
// CHECK-NOT:   call i32 @__kmpc_single
// CHECK-NOT:   call void @__kmpc_end_single
  return a;
}

// CHECK-LABEL: parallel_single
void parallel_single(float *a) {
#pragma omp parallel
#pragma omp single
  // CHECK-NOT: __kmpc_global_thread_num
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += i;
}

#endif
