// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>

void FooBarBaz() {
  __sanitizer_print_stack_trace();
}

int main() {
  FooBarBaz();
  return 0;
}
// CHECK: {{    #0 0x.* in __sanitizer_print_stack_trace}}
// CHECK: {{    #1 0x.* in FooBarBaz\(\) .*print-stack-trace.cc:7}}
// CHECK: {{    #2 0x.* in main .*print-stack-trace.cc:11}}
