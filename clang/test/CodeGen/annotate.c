// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

__attribute((annotate("foo"))) char foo;
void a(char *a) { 
  __attribute__((annotate("bar"))) static char bar;
}

// CHECK: @llvm.global.annotations = appending global [2 x %0]
