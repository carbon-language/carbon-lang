// no PCH
// RUN: %clang_cc1 -fopenmp=libiomp5 -emit-llvm -include %s -include %s %s -o - | FileCheck %s
// with PCH
// RUN: %clang_cc1 -fopenmp=libiomp5 -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s
#if !defined(PASS1)
#define PASS1

extern "C" int* malloc (int size);
int *a = malloc(20);

#elif !defined(PASS2)
#define PASS2

#pragma omp threadprivate(a)

#else

// CHECK: call {{.*}} @__kmpc_threadprivate_register(

// CHECK-LABEL: foo
int foo() {
  return *a;
  // CHECK: call {{.*}} @__kmpc_global_thread_num(
  // CHECK: call {{.*}} @__kmpc_threadprivate_cached(
}
#endif
