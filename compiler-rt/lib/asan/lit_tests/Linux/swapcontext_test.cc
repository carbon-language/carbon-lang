// Check that ASan plays well with easy cases of makecontext/swapcontext.

// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O1 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O2 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O3 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O1 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O2 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O3 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <ucontext.h>
#include <unistd.h>

ucontext_t orig_context;
ucontext_t child_context;

void Child(int mode) {
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child: %p\n", x);
  // (a) Do nothing, just return to parent function.
  // (b) Jump into the original function. Stack remains poisoned unless we do
  //     something.
  if (mode == 1) {
    if (swapcontext(&child_context, &orig_context) < 0) {
      perror("swapcontext");
      _exit(0);
    }
  }
}

int Run(int arg, int mode) {
  const int kStackSize = 1 << 20;
  char child_stack[kStackSize + 1];
  printf("Child stack: %p\n", child_stack);
  // Setup child context.
  getcontext(&child_context);
  child_context.uc_stack.ss_sp = child_stack;
  child_context.uc_stack.ss_size = kStackSize / 2;
  if (mode == 0) {
    child_context.uc_link = &orig_context;
  }
  makecontext(&child_context, (void (*)())Child, 1, mode);
  if (swapcontext(&orig_context, &child_context) < 0) {
    perror("swapcontext");
    return 0;
  }
  // Touch childs's stack to make sure it's unpoisoned.
  for (int i = 0; i < kStackSize; i++) {
    child_stack[i] = i;
  }
  return child_stack[arg];
}

int main(int argc, char **argv) {
  // CHECK: WARNING: ASan doesn't fully support makecontext/swapcontext
  int ret = 0;
  ret += Run(argc - 1, 0);
  printf("Test1 passed\n");
  // CHECK: Test1 passed
  ret += Run(argc - 1, 1);
  printf("Test2 passed\n");
  // CHECK: Test2 passed
  return ret;
}
