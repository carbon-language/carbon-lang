// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -g -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
//
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-LABEL: define {{.*void}} @{{.*}}without_schedule_clause{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void without_schedule_clause(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// UB = min(UB, GlobalUB)
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4571423
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 4571423, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// Loop header
// CHECK: [[IV:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (int i = 33; i < 32000000; i += 7) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 7
// CHECK-NEXT: [[CALC_I_2:%.+]] = add nsw i32 33, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK: call {{.+}} @__kmpc_cancel_barrier([[IDENT_T_TY]]* [[DEFAULT_LOC_BARRIER:[@%].+]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_not_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_not_chunked(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(static)
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// UB = min(UB, GlobalUB)
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4571423
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 4571423, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// Loop header
// CHECK: [[IV:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (int i = 32000000; i > 33; i += -7) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 7
// CHECK-NEXT: [[CALC_I_2:%.+]] = sub nsw i32 32000000, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK: call {{.+}} @__kmpc_cancel_barrier([[IDENT_T_TY]]* [[DEFAULT_LOC_BARRIER:[@%].+]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_chunked(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(static, 5)
// CHECK: call void @__kmpc_for_static_init_4u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 33, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 5)
// UB = min(UB, GlobalUB)
// CHECK: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp ugt i32 [[UB]], 16908288
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 16908288, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]

// Outer loop header
// CHECK: [[O_IV:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[O_UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ule i32 [[O_IV]], [[O_UB]]
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[IV:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ule i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (unsigned i = 131071; i <= 2147483647; i += 127) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul i32 [[IV1_1]], 127
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i32 131071, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// Update the counters, adding stride
// CHECK:  [[LB:%.+]] = load i32* [[OMP_LB]]
// CHECK-NEXT: [[ST:%.+]] = load i32* [[OMP_ST]]
// CHECK-NEXT: [[ADD_LB:%.+]] = add i32 [[LB]], [[ST]]
// CHECK-NEXT: store i32 [[ADD_LB]], i32* [[OMP_LB]]
// CHECK-NEXT: [[UB:%.+]] = load i32* [[OMP_UB]]
// CHECK-NEXT: [[ST:%.+]] = load i32* [[OMP_ST]]
// CHECK-NEXT: [[ADD_UB:%.+]] = add i32 [[UB]], [[ST]]
// CHECK-NEXT: store i32 [[ADD_UB]], i32* [[OMP_UB]]

// CHECK: [[O_LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK: call {{.+}} @__kmpc_cancel_barrier([[IDENT_T_TY]]* [[DEFAULT_LOC_BARRIER:[@%].+]], i32 [[GTID]])
// CHECK: ret void
}

void parallel_for(float *a) {
#pragma omp parallel
#pragma omp for schedule(static, 5)
  // CHECK-NOT: __kmpc_global_thread_num
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += i;
}

#endif // HEADER

