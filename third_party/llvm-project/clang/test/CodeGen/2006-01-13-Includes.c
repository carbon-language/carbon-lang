// RUN: %clang_cc1 %s -debug-info-kind=limited -emit-llvm -o - | FileCheck %s
// PR676

int printf(const char * restrict format, ...);

void test() {
  printf("Hello World\n");
}

// CHECK: test{{[\\/]}}CodeGen
