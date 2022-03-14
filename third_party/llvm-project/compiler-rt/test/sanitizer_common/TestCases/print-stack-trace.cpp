// RUN: %clangxx -O0 %s -o %t && %env_tool_opts=stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O3 %s -o %t && %env_tool_opts=stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=stack_trace_format=frame%n_lineno%l %run %t 2>&1 | FileCheck %s --check-prefix=CUSTOM
// RUN: %env_tool_opts=symbolize_inline_frames=false:stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s --check-prefix=NOINLINE
// RUN: %env_tool_opts=stack_trace_format='"frame:%n address:%%p"' %run %t 2>&1 | FileCheck %s --check-prefix=NOSYMBOLIZE

// UNSUPPORTED: darwin

// TODO(yln): temporary failing due to refactoring
// UNSUPPORTED: ubsan

#include <sanitizer/common_interface_defs.h>

static inline void FooBarBaz() {
  __sanitizer_print_stack_trace();
}

int main() {
  FooBarBaz();
  return 0;
}
// CHECK: {{    #0 0x.* in __sanitizer_print_stack_trace}}
// CHECK: {{    #1 0x.* in FooBarBaz(\(\))? .*}}print-stack-trace.cpp:[[@LINE-8]]
// CHECK: {{    #2 0x.* in main.*}}print-stack-trace.cpp:[[@LINE-5]]

// CUSTOM: frame1_lineno[[@LINE-11]]
// CUSTOM: frame2_lineno[[@LINE-8]]

// NOINLINE: #0 0x{{.*}} in __sanitizer_print_stack_trace
// NOINLINE: #1 0x{{.*}} in main{{.*}}print-stack-trace.cpp:[[@LINE-15]]

// NOSYMBOLIZE: frame:0 address:{{0x.*}}
// NOSYMBOLIZE: frame:1 address:{{0x.*}}
// NOSYMBOLIZE: frame:2 address:{{0x.*}}
