// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  S<T> &operator=(const S<T> &);
  operator T() { return T(); }
  ~S() {}
};

volatile int g = 1212;

// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[CAP_MAIN_TY:%.+]] = type { i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]* }
// CHECK: [[S_INT_TY:%.+]] = type { i32 }
// CHECK: [[CAP_TMAIN_TY:%.+]] = type { i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]* }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-DAG: [[SINGLE_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 322, i32 0, i32 0, i8*
// CHECK-DAG: [[SECTIONS_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 194, i32 0, i32 0, i8*
// CHECK-DAG: [[X:@.+]] = global double 0.0
template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp parallel
#pragma omp sections lastprivate(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
#pragma omp section
    s_arr[0] = var;
  }
  return T();
}

namespace A {
double x;
}
namespace B {
using A::x;
}

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i8* %{{.+}})
#pragma omp parallel
#pragma omp sections lastprivate(g)
  {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* %{{.+}}, i32* %{{.+}}, %{{.+}}* [[ARG:%.+]])
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store %{{.+}}* [[ARG]], %{{.+}}** [[ARG_REF:%.+]],
    // LAMBDA: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // LAMBDA: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
    // LAMBDA: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // LAMBDA: store volatile i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_PRIVATE_ADDR]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    // LAMBDA: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g = 1;
    // Check for final copying of private values back to original vars.
    // LAMBDA: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
    // LAMBDA: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
    // LAMBDA: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
    // LAMBDA: [[LAST_THEN]]
    // Actual copying.

    // original g=private_g;
    // LAMBDA: [[G_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // LAMBDA: store volatile i{{[0-9]+}} [[G_VAL]], i{{[0-9]+}}* [[G]],
    // LAMBDA: br label %[[LAST_DONE]]
    // LAMBDA: [[LAST_DONE]]
    // LAMBDA: call i32 @__kmpc_cancel_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
#pragma omp section
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i8* %{{.+}})
#pragma omp parallel
#pragma omp sections lastprivate(g)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* %{{.+}}, i32* %{{.+}}, %{{.+}}* [[ARG:%.+]])
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: store %{{.+}}* [[ARG]], %{{.+}}** [[ARG_REF:%.+]],
    // BLOCKS: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // BLOCKS: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
    // BLOCKS: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // BLOCKS: store volatile i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    // BLOCKS: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g = 1;
    // Check for final copying of private values back to original vars.
    // BLOCKS: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
    // BLOCKS: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
    // BLOCKS: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
    // BLOCKS: [[LAST_THEN]]
    // Actual copying.

    // original g=private_g;
    // BLOCKS: [[G_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // BLOCKS: store volatile i{{[0-9]+}} [[G_VAL]], i{{[0-9]+}}* [[G]],
    // BLOCKS: br label %[[LAST_DONE]]
    // BLOCKS: [[LAST_DONE]]
    // BLOCKS: call i32 @__kmpc_cancel_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
#pragma omp section
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp parallel
#pragma omp sections lastprivate(t_var, vec, s_arr, var)
  {
    {
    vec[0] = t_var;
    s_arr[0] = var;
    }
  }
#pragma omp parallel
#pragma omp sections lastprivate(A::x, B::x)
  {
    A::x++;
#pragma omp section
    ;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: %{{.+}} = bitcast [[CAP_MAIN_TY]]*
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[CAP_MAIN_TY]]*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, %{{.+}}*)* [[MAIN_MICROTASK1:@.+]] to void
// CHECK: = call {{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, [[CAP_MAIN_TY]]* %{{.+}})
// CHECK-NOT: alloca i{{[0-9]+}},
// CHECK-NOT: alloca [2 x i{{[0-9]+}}],
// CHECK-NOT: alloca [2 x [[S_FLOAT_TY]]],
// CHECK-NOT: alloca [[S_FLOAT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call i32 @__kmpc_single(

// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK-DAG: getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3

// <Skip loop body>

// CHECK-NOT: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-NOT: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*

// CHECK: call void @__kmpc_end_single(

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[SINGLE_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

//
// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, %{{.+}}* %{{.+}})
// CHECK: [[X_PRIV:%.+]] = alloca double,
// CHECK-NOT: alloca double

// Check for default initialization.
// CHECK-NOT: [[X_PRIV]]

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original x=private_x;
// CHECK: [[X_VAL:%.+]] = load double, double* [[X_PRIV]],
// CHECK: store double [[X_VAL]], double* [[X]],
// CHECK-NEXT: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[SECTIONS_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[CAP_TMAIN_TY]]*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, [[CAP_TMAIN_TY]]* %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// Check for default initialization.
// CHECK: [[T_VAR_PTR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[T_VAR_PTR_REF]],
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK: [[VEC_PTR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_PTR_REF:%.+]],
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: [[S_ARR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** [[S_ARR_REF_PTR]],
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK: [[VAR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[VAR_REF_PTR]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original t_var=private_t_var;
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_REF]],

// original vec[]=private_vec[];
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST]], i8* [[VEC_SRC]],

// original s_arr[]=private_s_arr[];
// CHECK: [[S_ARR_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]] to [[S_INT_TY]]*
// CHECK: [[S_ARR_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_BEGIN]], [[S_ARR_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %[[S_ARR_BODY_DONE]], label %[[S_ARR_BODY]]
// CHECK: [[S_ARR_BODY_DONE]]

// original var=private_var;
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* [[VAR_REF]], [[S_INT_TY]]* {{.*}} [[VAR_PRIV]])
// CHECK: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[SECTIONS_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void
#endif

