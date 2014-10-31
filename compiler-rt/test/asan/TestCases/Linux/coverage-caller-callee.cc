// Test caller-callee coverage with large number of threads
// and various numbers of callers and callees.

// RUN: %clangxx_asan -mllvm -asan-coverage=4 %s -o %t
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 10 1 2>&1 | FileCheck %s --check-prefix=CHECK-10-1
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 9  2 2>&1 | FileCheck %s --check-prefix=CHECK-9-2
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 7  3 2>&1 | FileCheck %s --check-prefix=CHECK-7-3
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 17 1 2>&1 | FileCheck %s --check-prefix=CHECK-17-1
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 15 2 2>&1 | FileCheck %s --check-prefix=CHECK-15-2
// RUN: ASAN_OPTIONS=coverage=1:verbosity=1 %run %t 18 3 2>&1 | FileCheck %s --check-prefix=CHECK-18-3
// RUN: rm -f caller-callee*.sancov
//
// REQUIRES: asan-64-bits
//
// CHECK-10-1: CovDump: 10 caller-callee pairs written
// CHECK-9-2: CovDump: 18 caller-callee pairs written
// CHECK-7-3: CovDump: 21 caller-callee pairs written
// CHECK-17-1: CovDump: 14 caller-callee pairs written
// CHECK-15-2: CovDump: 28 caller-callee pairs written
// CHECK-18-3: CovDump: 42 caller-callee pairs written

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
int P = 0;
struct Foo {virtual void f() {if (P) printf("Foo::f()\n");}};
struct Foo1 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo2 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo3 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo4 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo5 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo6 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo7 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo8 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo9 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo10 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo11 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo12 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo13 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo14 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo15 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo16 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo17 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo18 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo19 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};

Foo *foo[20] = {
    new Foo,   new Foo1,  new Foo2,  new Foo3,  new Foo4,  new Foo5,  new Foo6,
    new Foo7,  new Foo8,  new Foo9,  new Foo10, new Foo11, new Foo12, new Foo13,
    new Foo14, new Foo15, new Foo16, new Foo17, new Foo18, new Foo19,
};

int n_functions = 10;
int n_callers = 2;

void *Thread(void *arg) {
  if (n_callers >= 1) for (int i = 0; i < 2000; i++) foo[i % n_functions]->f();
  if (n_callers >= 2) for (int i = 0; i < 2000; i++) foo[i % n_functions]->f();
  if (n_callers >= 3) for (int i = 0; i < 2000; i++) foo[i % n_functions]->f();
  return arg;
}

int main(int argc, char **argv) {
  if (argc >= 2)
    n_functions = atoi(argv[1]);
  if (argc >= 3)
    n_callers = atoi(argv[2]);
  const int kNumThreads = 16;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++)
    pthread_create(&t[i], 0, Thread, 0);
  for (int i = 0; i < kNumThreads; i++)
    pthread_join(t[i], 0);
}
