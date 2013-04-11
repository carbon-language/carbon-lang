// RUN: %clangxx_asan -fsanitize=use-after-return -m64 -O0 %s -o %t && \
// RUN:   %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -m64 -O2 %s -o %t && \
// RUN:   %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -m32 -O2 %s -o %t && \
// RUN:   %t 2>&1 | %symbolize | FileCheck %s

#include <stdio.h>
#include <string.h>

__attribute__((noinline))
inline char *pretend_to_do_something(char *x) {
  __asm__ __volatile__("" : : "r" (x) : "memory");
  return x;
}

__attribute__((noinline))
char *LeakStack() {
  char x[1024];
  memset(x, 0, sizeof(x));
  return pretend_to_do_something(x);
}

__attribute__((noinline))
void RecuriveFunctionWithStackFrame(int depth) {
  if (depth <= 0) return;
  char x[1000];
  memset(x, 0, sizeof(x));
  pretend_to_do_something(x);
  RecuriveFunctionWithStackFrame(depth - 1);
  memset(x, 0, sizeof(x));
}

int main(int argc, char **argv) {
  char *stale_stack = LeakStack();
  RecuriveFunctionWithStackFrame(10);
  RecuriveFunctionWithStackFrame(20);
  RecuriveFunctionWithStackFrame(30);
  stale_stack[100]++;
  // CHECK: ERROR: AddressSanitizer: stack-use-after-return on address
  // CHECK: is located in stack of thread T0 at offset 132 in frame
  // CHECK:  in LeakStack(){{.*}}heavy_uar_test.cc:
  // CHECK: [32, 1056) 'x'
  return 0;
}
