// RUN: %clangxx -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// Not yet implemented for TSan.
// https://code.google.com/p/address-sanitizer/issues/detail?id=243
// XFAIL: tsan,lsan

#include <sanitizer/common_interface_defs.h>

void FooBarBaz() {
  __sanitizer_print_stack_trace();
}

int main() {
  FooBarBaz();
  return 0;
}
// CHECK: {{    #0 0x.* in __sanitizer_print_stack_trace}}
// CHECK: {{    #1 0x.* in FooBarBaz\(\) .*print-stack-trace.cc:11}}
// CHECK: {{    #2 0x.* in main.*print-stack-trace.cc:15}}
