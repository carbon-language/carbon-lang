// Test for the leak_check_at_exit flag.
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS="verbosity=1" %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: LSAN_OPTIONS="verbosity=1" %t 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: LSAN_OPTIONS="verbosity=1:leak_check_at_exit=0" ASAN_OPTIONS="$ASAN_OPTIONS:leak_check_at_exit=0" %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-do
// RUN: LSAN_OPTIONS="verbosity=1:leak_check_at_exit=0" ASAN_OPTIONS="$ASAN_OPTIONS:leak_check_at_exit=0" %t 2>&1 | FileCheck %s --check-prefix=CHECK-dont

#include <stdio.h>
#include <sanitizer/lsan_interface.h>

int main(int argc, char *argv[]) {
  printf("printf to break optimization\n");
  if (argc > 1)
    __lsan_do_leak_check();
  return 0;
}

// CHECK-do: SUMMARY: LeakSanitizer:
// CHECK-dont-NOT: SUMMARY: LeakSanitizer:
