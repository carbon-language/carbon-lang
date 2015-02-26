// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK:       [[UNNAMED_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// CHECK:       [[THE_NAME_LOCK:@.+]] = common global [8 x i32] zeroinitializer

// CHECK:       define void [[FOO:@.+]]()

void foo() {}

// CHECK-LABEL: @main
int main() {
// CHECK:       [[A_ADDR:%.+]] = alloca i8
  char a;

// CHECK:       [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       call void @__kmpc_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[UNNAMED_LOCK]])
// CHECK-NEXT:  store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  call void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[UNNAMED_LOCK]])
#pragma omp critical
  a = 2;
// CHECK:       call void @__kmpc_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
// CHECK-NEXT:  call void [[FOO]]()
// CHECK-NEXT:  call void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
#pragma omp critical(the_name)
  foo();
// CHECK-NOT:   call void @__kmpc_critical
// CHECK-NOT:   call void @__kmpc_end_critical
  return a;
}

// CHECK-LABEL: parallel_critical
void parallel_critical(float *a) {
#pragma omp parallel
#pragma omp critical
  // CHECK-NOT: __kmpc_global_thread_num
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += i;
}

#endif
