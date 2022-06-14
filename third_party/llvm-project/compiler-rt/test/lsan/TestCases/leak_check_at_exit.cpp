// Test for the leak_check_at_exit flag.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 not %run %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0:leak_check_at_exit=0 not %run %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0:leak_check_at_exit=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-dont

#include <stdio.h>
#include <stdlib.h>
#include <sanitizer/lsan_interface.h>

int main(int argc, char *argv[]) {
  fprintf(stderr, "Test alloc: %p.\n", malloc(1337));
  if (argc > 1)
    __lsan_do_leak_check();
  return 0;
}

// CHECK-do: SUMMARY: {{(Leak|Address)}}Sanitizer:
// CHECK-dont-NOT: SUMMARY: {{(Leak|Address)}}Sanitizer:
