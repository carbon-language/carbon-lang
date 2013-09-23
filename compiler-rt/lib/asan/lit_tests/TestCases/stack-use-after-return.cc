// RUN: export ASAN_OPTIONS=detect_stack_use_after_return=1
// RUN: %clangxx_asan -fsanitize=use-after-return -O0 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O1 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O2 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O3 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=detect_stack_use_after_return=0 %t
// Regression test for a CHECK failure with small stack size and large frame.
// RUN: %clangxx_asan -fsanitize=use-after-return -O3 %s -o %t -DkSize=10000 && \
// RUN: (ulimit -s 65;  not %t) 2>&1 | FileCheck %s
//
// Test that we can find UAR in a thread other than main:
// RUN: %clangxx_asan -fsanitize=use-after-return -DUseThread -O2 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck --check-prefix=THREAD %s

#include <stdio.h>
#include <pthread.h>

#ifndef kSize
# define kSize 1
#endif

#ifndef UseThread
# define UseThread 0
#endif

__attribute__((noinline))
char *Ident(char *x) {
  fprintf(stderr, "1: %p\n", x);
  return x;
}

__attribute__((noinline))
char *Func1() {
  char local[kSize];
  return Ident(local);
}

__attribute__((noinline))
void Func2(char *x) {
  fprintf(stderr, "2: %p\n", x);
  *x = 1;
  // CHECK: WRITE of size 1 {{.*}} thread T0
  // CHECK:     #0{{.*}}Func2{{.*}}stack-use-after-return.cc:[[@LINE-2]]
  // CHECK: is located in stack of thread T0 at offset
  // CHECK: 'local' <== Memory access at offset 32 is inside this variable
  // THREAD: WRITE of size 1 {{.*}} thread T{{[1-9]}}
  // THREAD:     #0{{.*}}Func2{{.*}}stack-use-after-return.cc:[[@LINE-6]]
  // THREAD: is located in stack of thread T{{[1-9]}} at offset
  // THREAD: 'local' <== Memory access at offset 32 is inside this variable
}

void *Thread(void *unused)  {
  Func2(Func1());
  return NULL;
}

int main(int argc, char **argv) {
#if UseThread
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
#else
  Func2(Func1());
#endif
  return 0;
}
