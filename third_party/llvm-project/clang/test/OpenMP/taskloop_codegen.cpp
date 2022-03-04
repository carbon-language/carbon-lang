// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=50 -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=50 -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: @main
int main(int argc, char **argv) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEFLOC:@.+]])
// CHECK: call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]],
// CHECK: call i32 @__kmpc_omp_task(%struct.ident_t* [[DEFLOC]], i32 [[GTID]],
#pragma omp task
  ;
// CHECK: call void @__kmpc_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 33, i64 80, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK1:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 0, i64 0, i8* null)
// CHECK: call void @__kmpc_end_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
#pragma omp taskloop priority(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK2:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[GRAINSIZE:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 1, i64 [[GRAINSIZE]], i8* null)
#pragma omp taskloop nogroup grainsize(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call void @__kmpc_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK3:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[IF:%.+]] = icmp ne i32 %{{.+}}, 0
// CHECK: [[IF_INT:%.+]] = sext i1 [[IF]] to i32
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 [[IF_INT]], i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 4, i8* null)
// CHECK: call void @__kmpc_end_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
  int i;
#pragma omp taskloop if(argc) shared(argc, argv) collapse(2) num_tasks(4)
  for (i = 0; i < argc; ++i)
  for (int j = argc; j < argv[argc][argc]; ++j)
    ;
// CHECK: call void @__kmpc_taskgroup(
// CHECK: call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 1, i64 80, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{.+}}*)* [[TASK_CANCEL:@.+]] to i32 (i32, i8*)*))
// CHECK: call void @__kmpc_taskloop(
// CHECK: call void @__kmpc_end_taskgroup(
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
  }
}

// CHECK: define internal noundef i32 [[TASK1]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK2]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK3]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[CNT:%.+]],
// CHECK: br label
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK_CANCEL]](
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[IS_CANCEL]], label %[[EXIT:.+]], label %[[CONTINUE:[^,]+]]
// CHECK: [[EXIT]]:
// CHECK: store i32 1, i32* [[CLEANUP_SLOT:%.+]],
// CHECK: br label %[[DONE:[^,]+]]
// CHECK: [[CONTINUE]]:
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[IS_CANCEL]], label %[[EXIT2:.+]], label %[[CONTINUE2:[^,]+]]
// CHECK: [[EXIT2]]:
// CHECK: store i32 1, i32* [[CLEANUP_SLOT]],
// CHECK: br label %[[DONE]]
// CHECK: store i32 0, i32* [[CLEANUP_SLOT]],
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: ret i32 0

// CHECK-LABEL: @_ZN1SC2Ei
struct S {
  int a;
  S(int c) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEFLOC:@.+]])
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK4:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[NUM_TASKS:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 [[NUM_TASKS]], i8* null)
#pragma omp taskloop shared(c) num_tasks(a)
    for (a = 0; a < c; ++a)
      ;
  }
} s(1);

// CHECK: define internal noundef i32 [[TASK4]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

class St {
public:
  operator int();
  St &operator+=(int);
};

// CHECK-LABEL: taskloop_with_class
void taskloop_with_class() {
  St s1;
  // CHECK: [[TD:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 [[GTID:%.+]], i32 1, i64 88, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, [[TD_TYPE:%.+]]*)* @{{.+}} to i32 (i32, i8*)*))
  // CHECK: call void @__kmpc_taskloop(%struct.ident_t* @{{.+}}, i32 [[GTID]], i8* [[TD]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* bitcast (void ([[TD_TYPE]]*, [[TD_TYPE]]*, i32)* @{{.+}} to i8*))
#pragma omp taskloop
  for (St s = St(); s < s1; s += 1) {
  }
}

#endif
