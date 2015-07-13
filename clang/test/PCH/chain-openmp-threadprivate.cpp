// no PCH
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -emit-llvm -include %s -include %s %s -o - | FileCheck %s
// with PCH
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s
// no PCH
// RUN: %clang_cc1 -fopenmp -emit-llvm -include %s -include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS
// with PCH
// RUN: %clang_cc1 -fopenmp -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS
#if !defined(PASS1)
#define PASS1

extern "C" int* malloc (int size);
int *a = malloc(20);

#elif !defined(PASS2)
#define PASS2

#pragma omp threadprivate(a)

#else

// CHECK: call {{.*}} @__kmpc_threadprivate_register(
// CHECK-TLS: @a = {{.*}}thread_local {{.*}}global {{.*}}i32*

// CHECK-LABEL: foo
// CHECK-TLS-LABEL: foo
int foo() {
  return *a;
  // CHECK: call {{.*}} @__kmpc_global_thread_num(
  // CHECK: call {{.*}} @__kmpc_threadprivate_cached(
  // CHECK-TLS: call {{.*}}i32** @_ZTW1a()
}

// CHECK-TLS: define {{.*}}i32** @_ZTW1a()

#endif
