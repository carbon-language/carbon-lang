// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s
// expected-no-diagnostics
#ifndef ARRAY
#ifndef HEADER
#define HEADER

volatile int g = 1212;
#pragma omp threadprivate(g)

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  S &operator=(const S &) { return *this; };
  operator T() { return T(); }
  ~S() {}
};

// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*


// CHECK-DAG: [[T_VAR:@.+]] = internal global i{{[0-9]+}} 1122,
// CHECK-DAG: [[VEC:@.+]] = internal global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 1, i{{[0-9]+}} 2],
// CHECK-DAG: [[S_ARR:@.+]] = internal global [2 x [[S_FLOAT_TY]]] zeroinitializer,
// CHECK-DAG: [[VAR:@.+]] = internal global [[S_FLOAT_TY]] zeroinitializer,
// CHECK-DAG: [[TMAIN_T_VAR:@.+]] = linkonce_odr global i{{[0-9]+}} 333,
// CHECK-DAG: [[TMAIN_VEC:@.+]] = linkonce_odr global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 3, i{{[0-9]+}} 3],
// CHECK-DAG: [[TMAIN_S_ARR:@.+]] = linkonce_odr global [2 x [[S_INT_TY]]] zeroinitializer,
// CHECK-DAG: [[TMAIN_VAR:@.+]] = linkonce_odr global [[S_INT_TY]] zeroinitializer,
template <typename T>
T tmain() {
  S<T> test;
  test = S<T>();
  static T t_var = 333;
  static T vec[] = {3, 3};
  static S<T> s_arr[] = {1, 2};
  static S<T> var(3);
#pragma omp threadprivate(t_var, vec, s_arr, var)
#pragma omp parallel copyin(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
#pragma omp parallel copyin(t_var)
  {}
  return T();
}

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{( x86_thiscallcc)?}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i8*
#pragma omp parallel copyin(g)
  {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* %{{.+}}, i32* %{{.+}}, %{{.+}}* [[ARG:%.+]])

    // threadprivate_g = g;
    // LAMBDA: call i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
    // LAMBDA: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
    // LAMBDA: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[G]] to i{{[0-9]+}}), %{{.+}}
    // LAMBDA: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // LAMBDA: [[NOT_MASTER]]
    // LAMBDA: load i{{[0-9]+}}, i{{[0-9]+}}* [[G]],
    // LAMBDA: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
    // LAMBDA: [[DONE]]

    // LAMBDA: call i32 @__kmpc_cancel_barrier(
    g = 1;
    // LAMBDA: call{{( x86_thiscallcc)?}} void [[INNER_LAMBDA:@.+]](%{{.+}}*
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
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
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i8*
#pragma omp parallel copyin(g)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* %{{.+}}, i32* %{{.+}}, %{{.+}}* [[ARG:%.+]])

    // threadprivate_g = g;
    // BLOCKS: call i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
    // BLOCKS: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
    // BLOCKS: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[G]] to i{{[0-9]+}}), %{{.+}}
    // BLOCKS: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // BLOCKS: [[NOT_MASTER]]
    // BLOCKS: load i{{[0-9]+}}, i{{[0-9]+}}* [[G]],
    // BLOCKS: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
    // BLOCKS: [[DONE]]

    // BLOCKS: call i32 @__kmpc_cancel_barrier(
    g = 1;
    // BLOCKS: store volatile i{{[0-9]+}} 1, i{{[0-9]+}}*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: call i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
      // BLOCKS: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  test = S<float>();
  static int t_var = 1122;
  static int vec[] = {1, 2};
  static S<float> s_arr[] = {1, 2};
  static S<float> var(3);
#pragma omp threadprivate(t_var, vec, s_arr, var)
#pragma omp parallel copyin(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
#pragma omp parallel copyin(t_var)
  {}
  return tmain<int>();
#endif
}

// CHECK-LABEL: @main
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN:@.+]]([[S_FLOAT_TY]]* [[TEST]], [[S_FLOAT_TY]]*
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, {{%.+}}*)* [[MAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*), i8* %{{.+}})
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, {{%.+}}*)* [[MAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*), i8* %{{.+}})
// CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, {{%.+}}* %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// threadprivate_t_var = t_var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},

// threadprivate_vec = vec;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[VEC]]
// CHECK: call void @llvm.memcpy{{.*}}(i8* %{{.+}}, i8* bitcast ([2 x i{{[0-9]+}}]* [[VEC]] to i8*),

// threadprivate_s_arr = s_arr;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[S_ARR]]
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* {{%.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]

// threadprivate_var = var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[VAR]]
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{%.+}}, [[S_FLOAT_TY]]* {{.*}}[[VAR]])
// CHECK: [[DONE]]

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, {{%.+}}* %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// threadprivate_t_var = t_var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
// CHECK: [[DONE]]

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* [[TEST]], [[S_INT_TY]]*
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, {{%.+}}*)* [[TMAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*), i8* %{{.+}})
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, {{%.+}}*)* [[TMAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*), i8* %{{.+}})
// CHECK: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, {{%.+}}* %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// threadprivate_t_var = t_var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[TMAIN_T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},

// threadprivate_vec = vec;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_VEC]]
// CHECK: call void @llvm.memcpy{{.*}}(i8* %{{.+}}, i8* bitcast ([2 x i{{[0-9]+}}]* [[TMAIN_VEC]] to i8*),

// threadprivate_s_arr = s_arr;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_S_ARR]]
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* {{%.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]

// threadprivate_var = var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_VAR]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{%.+}}, [[S_INT_TY]]* {{.*}}[[TMAIN_VAR]])
// CHECK: [[DONE]]

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// CHECK: define internal void [[TMAIN_MICROTASK1]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, {{%.+}}* %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// threadprivate_t_var = t_var;
// CHECK: call i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[TMAIN_T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
// CHECK: [[DONE]]

// CHECK: call i32 @__kmpc_cancel_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

#endif
#else
// ARRAY-LABEL: array_func
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St &operator=(const St &) { return *this; };
  ~St() {}
};

void array_func() {
  static int a[2];
  static St s[2];
// ARRAY: @__kmpc_fork_call(
// ARRAY: call i8* @__kmpc_threadprivate_cached(
// ARRAY: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %{{.+}}, i8* bitcast ([2 x i32]* @{{.+}} to i8*), i64 8, i32 4, i1 false)
// ARRAY: call dereferenceable(8) %struct.St* @{{.+}}(%struct.St* %{{.+}}, %struct.St* dereferenceable(8) %{{.+}})
#pragma omp threadprivate(a, s)
#pragma omp parallel copyin(a, s)
  ;
}
#endif


