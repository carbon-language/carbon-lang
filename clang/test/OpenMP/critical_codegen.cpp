// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp=libiomp5 -fexceptions -fcxx-exceptions -gline-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK:       [[UNNAMED_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// CHECK:       [[THE_NAME_LOCK:@.+]] = common global [8 x i32] zeroinitializer

// CHECK:       define void [[FOO:@.+]]()

void foo() {}

// CHECK-LABEL: @main
// TERM_DEBUG-LABEL: @main
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
// CHECK-NEXT:  invoke void [[FOO]]()
// CHECK:       call void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
#pragma omp critical(the_name)
  foo();
// CHECK-NOT:   call void @__kmpc_critical
// CHECK-NOT:   call void @__kmpc_end_critical
  return a;
}

// CHECK-LABEL:      parallel_critical
// TERM_DEBUG-LABEL: parallel_critical
void parallel_critical() {
#pragma omp parallel
#pragma omp critical
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_critical({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_critical({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();
}
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !MDLocation(line: 44,
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !MDLocation(line: 44,
#endif
