// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &st) : a(st.a + st.b), b(0) {}
  ~St() {}
};

volatile int g;

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  S(const S &s, St t = St()) : f(s.f + t.a) {}
  operator T() { return T(); }
  ~S() {}
};

// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG: [[ST_TY:%.+]] = type { i{{[0-9]+}}, i{{[0-9]+}} }
// CHECK-DAG: [[CAP_MAIN_TY:%.+]] = type { [2 x i{{[0-9]+}}]*, i{{[0-9]+}}*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]* }
// CHECK-DAG: [[CAP_TMAIN_TY:%.+]] = type { [2 x i{{[0-9]+}}]*, i{{[0-9]+}}*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]* }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*

template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp parallel firstprivate(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp parallel firstprivate(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return tmain<int>();
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: %{{.+}} = bitcast [[CAP_MAIN_TY]]*
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...)* @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[CAP_MAIN_TY]]*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, [[CAP_MAIN_TY]]* %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[T_VAR_PTR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}** [[T_VAR_PTR_REF]],
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}* [[T_VAR_REF]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: [[VEC_PTR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]** [[VEC_PTR_REF:%.+]],
// CHECK: br label %[[VEC_PRIV_INIT:.+]]
// CHECK: [[VEC_PRIV_INIT]]
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST]], i8* [[VEC_SRC]],
// CHECK: br label %[[VEC_PRIV_INIT_END:.+]]
// CHECK: [[VEC_PRIV_INIT_END]]
// CHECK: [[S_ARR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_FLOAT_TY]]]** [[S_ARR_REF_PTR]],
// CHECK: br label %[[S_ARR_PRIV_INIT:.+]]
// CHECK: [[S_ARR_PRIV_INIT]]
// CHECK: [[S_ARR_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]]* [[S_ARR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_END:%.+]] = getelementptr [[S_FLOAT_TY]]* [[S_ARR_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR:@.+]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_CONSTR:@.+]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}}, [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR:@.+]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]
// CHECK: br label %[[S_ARR_PRIV_INIT_END:.+]]
// CHECK: [[S_ARR_PRIV_INIT_END]]
// CHECK: [[VAR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]** [[VAR_REF_PTR]],
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]], [[S_FLOAT_TY]]* {{.*}} [[VAR_REF]], [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...)* @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[CAP_TMAIN_TY]]*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* [[GTID_ADDR:%.+]], i{{[0-9]+}}* %{{.+}}, [[CAP_TMAIN_TY]]* %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[T_VAR_PTR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}** [[T_VAR_PTR_REF]],
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}* [[T_VAR_REF]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: [[VEC_PTR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]** [[VEC_PTR_REF:%.+]],
// CHECK: br label %[[VEC_PRIV_INIT:.+]]
// CHECK: [[VEC_PRIV_INIT]]
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST]], i8* [[VEC_SRC]],
// CHECK: br label %[[VEC_PRIV_INIT_END:.+]]
// CHECK: [[VEC_PRIV_INIT_END]]
// CHECK: [[S_ARR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_INT_TY]]]** [[S_ARR_REF_PTR]],
// CHECK: br label %[[S_ARR_PRIV_INIT:.+]]
// CHECK: [[S_ARR_PRIV_INIT]]
// CHECK: [[S_ARR_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]]* [[S_ARR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_END:%.+]] = getelementptr [[S_INT_TY]]* [[S_ARR_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR:@.+]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}}, [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]
// CHECK: br label %[[S_ARR_PRIV_INIT_END:.+]]
// CHECK: [[S_ARR_PRIV_INIT_END]]
// CHECK: [[VAR_REF_PTR:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]** [[VAR_REF_PTR]],
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]], [[S_INT_TY]]* {{.*}} [[VAR_REF]], [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: ret void
#endif

