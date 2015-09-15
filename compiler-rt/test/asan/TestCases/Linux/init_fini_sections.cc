// RUN: %clangxx_asan %s -o %t && %run %t | FileCheck %s

#include <stdio.h>

static void foo() {
  printf("foo\n");
}

int main() {
  return 0;
}

__attribute__((section(".preinit_array")))
void (*call_foo)(void) = &foo;

__attribute__((section(".init_array")))
void (*call_foo_2)(void) = &foo;

__attribute__((section(".fini_array")))
void (*call_foo_3)(void) = &foo;

// CHECK: foo
// CHECK: foo
// CHECK: foo
