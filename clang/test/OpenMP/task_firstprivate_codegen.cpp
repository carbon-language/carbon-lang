// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s
// expected-no-diagnostics

// It doesn't pass on win32.
// REQUIRES: shell
#ifndef ARRAY
#ifndef HEADER
#define HEADER

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  S(const S &s, T t = T()) : f(s.f + t) {}
  operator T() { return T(); }
  ~S() {}
};

volatile double g;

// CHECK-DAG: [[KMP_TASK_T_TY:%.+]] = type { i8*, i32 (i32, i8*)*, i32, i32 (i32, i8*)* }
// CHECK-DAG: [[S_DOUBLE_TY:%.+]] = type { double }
// CHECK-DAG: [[CAP_MAIN_TY:%.+]] = type { [2 x i32]*, i32*, [2 x [[S_DOUBLE_TY]]]*, [[S_DOUBLE_TY]]* }
// CHECK-DAG: [[PRIVATES_MAIN_TY:%.+]] = type {{.?}}{ [[S_DOUBLE_TY]], [2 x [[S_DOUBLE_TY]]], i32, [2 x i32]
// CHECK-DAG: [[KMP_TASK_MAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [[PRIVATES_MAIN_TY]] }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[CAP_TMAIN_TY:%.+]] = type { [2 x i32]*, i32*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]* }
// CHECK-DAG: [[PRIVATES_TMAIN_TY:%.+]] = type { i32, [2 x i32], [2 x [[S_INT_TY]]], [[S_INT_TY]] }
// CHECK-DAG: [[KMP_TASK_TMAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [[PRIVATES_TMAIN_TY]] }
template <typename T>
T tmain() {
  S<T> ttt;
  S<T> test(ttt);
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp task firstprivate(t_var, vec, s_arr, s_arr, var, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global double
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{( x86_thiscallcc)?}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 40, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// LAMBDA: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// LAMBDA: [[G_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_ADDR_REF]]
// LAMBDA: [[G_VAL:%.+]] = load volatile double, double* [[G_REF]]
// LAMBDA: store volatile double [[G_VAL]], double* [[G_PRIVATE_ADDR]]
// LAMBDA: [[G_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// LAMBDA: store double* [[G_PRIVATE_ADDR]], double** [[G_ADDR_REF]],
// LAMBDA: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
// LAMBDA: ret
#pragma omp task firstprivate(g)
  {
    // LAMBDA: define {{.+}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG_PTR:%.+]])
    // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
    // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
    // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
    // LAMBDA: store volatile double 2.0{{.+}}, double* [[G_REF]]

    // LAMBDA: define internal i32 [[TASK_ENTRY]](i32, %{{.+}}*)
    g = 1;
    // LAMBDA: store volatile double 1.0{{.+}}, double* %{{.+}},
    // LAMBDA: call void [[INNER_LAMBDA]](%
    // LAMBDA: ret
    [&]() {
      g = 2;
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global double
  // BLOCKS-LABEL: @main
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 40, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
  // BLOCKS: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
  // BLOCKS: [[G_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 0
  // BLOCKS: [[G_REF:%.+]] = load double*, double** [[G_ADDR_REF]]
  // BLOCKS: [[G_VAL:%.+]] = load volatile double, double* [[G_REF]]
  // BLOCKS: store volatile double [[G_VAL]], double* [[G_PRIVATE_ADDR]]
  // BLOCKS: [[G_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 0
  // BLOCKS: store double* [[G_PRIVATE_ADDR]], double** [[G_ADDR_REF]],
  // BLOCKS: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  // BLOCKS: ret
#pragma omp task firstprivate(g)
  {
    // BLOCKS: define {{.+}} void {{@.+}}(i8*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store volatile double 2.0{{.+}}, double*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: ret

    // BLOCKS: define internal i32 [[TASK_ENTRY]](i32, %{{.+}}*)
    g = 1;
    // BLOCKS: store volatile double 1.0{{.+}}, double* %{{.+}},
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    ^{
      g = 2;
    }();
  }
  }();
  return 0;
#else
  S<double> ttt;
  S<double> test(ttt);
  int t_var = 0;
  int vec[] = {1, 2};
  S<double> s_arr[] = {1, 2};
  S<double> var(3);
#pragma omp task firstprivate(var, t_var, s_arr, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: alloca [[S_DOUBLE_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[LOC:%.+]])

// CHECK: call {{.*}} [[S_DOUBLE_TY_COPY_CONSTR:@.+]]([[S_DOUBLE_TY]]* [[TEST]],

// Store original variables in capture struct.
// CHECK: [[VEC_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: store [2 x i32]* [[VEC_ADDR]], [2 x i32]** [[VEC_REF]],
// CHECK: [[T_VAR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store i32* [[T_VAR_ADDR]], i32** [[T_VAR_REF]],
// CHECK: [[S_ARR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: store [2 x [[S_DOUBLE_TY]]]* [[S_ARR_ADDR]], [2 x [[S_DOUBLE_TY]]]** [[S_ARR_REF]],
// CHECK: [[VAR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: store [[S_DOUBLE_TY]]* [[VAR_ADDR]], [[S_DOUBLE_TY]]** [[VAR_REF]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T]] task_data;
//         [[KMP_TASK_MAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc([[LOC]], i32 [[GTID]], i32 1, i64 72, i64 32, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_MAIN_TY]]*

// Fill kmp_task_t->shareds by copying from original capture argument.
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF_ADDR:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_ADDR]],
// CHECK: [[CAPTURES_ADDR:%.+]] = bitcast [[CAP_MAIN_TY]]* %{{.+}} to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[SHAREDS_REF]], i8* [[CAPTURES_ADDR]], i64 32, i32 8, i1 false)

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// Also copy address of private copy to the corresponding shareds reference.
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SHAREDS:%.+]] = bitcast i8* [[SHAREDS_REF]] to [[CAP_MAIN_TY]]*

// Constructors for s_arr and var.
// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[VAR_REF:%.+]] = load [[S_DOUBLE_TY]]*, [[S_DOUBLE_TY]]** [[VAR_ADDR_REF]],
// CHECK: call void [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]* [[PRIVATE_VAR_REF]], [[S_DOUBLE_TY]]* {{.*}}[[VAR_REF]],

// shareds->var_addr = &kmp_task_t->privates.var;
// CHECK: [[VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 3
// CHECK: store [[S_DOUBLE_TY]]* [[PRIVATE_VAR_REF]], [[S_DOUBLE_TY]]** [[VAR_ADDR_REF]],

// s_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[S_ARR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 2
// CHECK: load [2 x [[S_DOUBLE_TY]]]*, [2 x [[S_DOUBLE_TY]]]** [[S_ARR_ADDR_REF]],
// CHECK: call void [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]* [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// shareds->s_arr_addr = &kmp_task_t->privates.s_arr;
// CHECK: [[S_ARR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store [2 x [[S_DOUBLE_TY]]]* [[PRIVATE_S_ARR_REF]], [2 x [[S_DOUBLE_TY]]]** [[S_ARR_ADDR_REF]],

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[T_VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[T_VAR_REF:%.+]] = load i{{.+}}*, i{{.+}}** [[T_VAR_ADDR_REF]],
// CHECK: [[T_VAR:%.+]] = load i{{.+}}, i{{.+}}* [[T_VAR_REF]],
// CHECK: store i32 [[T_VAR]], i32* [[PRIVATE_T_VAR_REF]],

// shareds->t_var_addr = &kmp_task_t->privates.t_var;
// CHECK: [[T_VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i32* [[PRIVATE_T_VAR_REF]], i32** [[T_VAR_ADDR_REF]],

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[VEC_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 0
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

// shareds->vec_addr = &kmp_task_t->privates.vec;
// CHECK: [[VEC_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 0
// CHECK: store [2 x i32]* [[PRIVATE_VEC_REF]], [2 x i32]** [[VEC_ADDR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task([[LOC]], i32 [[GTID]], i8* [[RES]])

// CHECK: = call i{{.+}} [[TMAIN_INT:@.+]]()

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_DOUBLE_TY_DESTR:@.+]]([[S_DOUBLE_TY]]*
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//
// CHECK: define internal i32 [[TASK_ENTRY]](i32, [[KMP_TASK_MAIN_TY]]*)

// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_ADDR_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_ADDR_REF]],
// CHECK: [[SHAREDS:%.+]] = bitcast i8* [[SHAREDS_REF]] to [[CAP_MAIN_TY]]*

// Privates actually are used.
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 2
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 3

// CHECK: ret

// CHECK: define internal i32 [[DESTRUCTORS]](i32, [[KMP_TASK_MAIN_TY]]*)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], [2 x [[S_DOUBLE_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_DOUBLE_TY_DESTR]]([[S_DOUBLE_TY]]* [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: call void [[S_DOUBLE_TY_DESTR]]([[S_DOUBLE_TY]]* [[PRIVATE_VAR_REF]])
// CHECK: ret i32

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: alloca [[S_INT_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[LOC:%.+]])

// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]],

// Store original variables in capture struct.
// CHECK: [[VEC_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: store [2 x i32]* [[VEC_ADDR]], [2 x i32]** [[VEC_REF]],
// CHECK: [[T_VAR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store i32* [[T_VAR_ADDR]], i32** [[T_VAR_REF]],
// CHECK: [[S_ARR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: store [2 x [[S_INT_TY]]]* [[S_ARR_ADDR]], [2 x [[S_INT_TY]]]** [[S_ARR_REF]],
// CHECK: [[VAR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: store [[S_INT_TY]]* [[VAR_ADDR]], [[S_INT_TY]]** [[VAR_REF]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T_TY]] task_data;
//         [[KMP_TASK_TMAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc([[LOC]], i32 [[GTID]], i32 1, i64 56, i64 32, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_TMAIN_TY]]*

// Fill kmp_task_t->shareds by copying from original capture argument.
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF_ADDR:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_ADDR]],
// CHECK: [[CAPTURES_ADDR:%.+]] = bitcast [[CAP_TMAIN_TY]]* %{{.+}} to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[SHAREDS_REF]], i8* [[CAPTURES_ADDR]], i64 32, i32 8, i1 false)

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SHAREDS:%.+]] = bitcast i8* [[SHAREDS_REF]] to [[CAP_TMAIN_TY]]*

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[T_VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[T_VAR_REF:%.+]] = load i{{.+}}*, i{{.+}}** [[T_VAR_ADDR_REF]],
// CHECK: [[T_VAR:%.+]] = load i{{.+}}, i{{.+}}* [[T_VAR_REF]],
// CHECK: store i32 [[T_VAR]], i32* [[PRIVATE_T_VAR_REF]],

// shareds->t_var_addr = &kmp_task_t->privates.t_var;
// CHECK: [[T_VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i32* [[PRIVATE_T_VAR_REF]], i32** [[T_VAR_ADDR_REF]],

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[VEC_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 0
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

// shareds->vec_addr = &kmp_task_t->privates.vec;
// CHECK: [[VEC_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 0
// CHECK: store [2 x i32]* [[PRIVATE_VEC_REF]], [2 x i32]** [[VEC_ADDR_REF]],

// Constructors for s_arr and var.
// a_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[S_ARR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 2
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: call void [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// shareds->s_arr_addr = &kmp_task_t->privates.s_arr;
// CHECK: [[S_ARR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], [2 x [[S_INT_TY]]]** [[S_ARR_ADDR_REF]],

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* [[PRIVATE_VAR_REF]],

// shareds->var_addr = &kmp_task_t->privates.var;
// CHECK: [[VAR_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* [[SHAREDS]], i{{.+}} 0, i{{.+}} 3
// CHECK: store [[S_INT_TY]]* [[PRIVATE_VAR_REF]], [[S_INT_TY]]** [[VAR_ADDR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task([[LOC]], i32 [[GTID]], i8* [[RES]])

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//
// CHECK: define internal i32 [[TASK_ENTRY]](i32, [[KMP_TASK_TMAIN_TY]]*)

// Substitute addresses of shared variables in capture struct by address of private copies from kmp_task_t.
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_ADDR_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_ADDR_REF]],
// CHECK: [[SHAREDS:%.+]] = bitcast i8* [[SHAREDS_REF]] to [[CAP_TMAIN_TY]]*

// Privates actually are used.
// CHECK-DAG: getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 2
// CHECK-DAG: getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{.+}} 0, i{{.+}} 3

// CHECK: ret

// CHECK: define internal i32 [[DESTRUCTORS]](i32, [[KMP_TASK_TMAIN_TY]]*)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

#endif
#else
// ARRAY-LABEL: array_func
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &) {}
  ~St() {}
};

void array_func(int n, float a[n], St s[2]) {
// ARRAY: call i8* @__kmpc_omp_task_alloc(
// ARRAY: store float** %{{.+}}, float*** %{{.+}},
// ARRAY: store %struct.St** %{{.+}}, %struct.St*** %{{.+}},
// ARRAY: call i32 @__kmpc_omp_task(
#pragma omp task firstprivate(a, s)
  ;
}
#endif

