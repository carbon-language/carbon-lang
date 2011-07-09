// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

__attribute((annotate("foo"))) char foo;
void a(char *a) { 
  __attribute__((annotate("bar"))) static char bar;
}

// CHECK: private unnamed_addr global
// CHECK: private unnamed_addr global
// CHECK: @llvm.global.annotations = appending global [2 x { i8*, i8*, i8*, i32 }]
