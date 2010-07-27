// RUN: %clang_cc1 -emit-pch -o %t1 %S/Inputs/chain-function1.h
// RUN: %clang_cc1 -emit-pch -o %t2 %S/Inputs/chain-function2.h -include-pch %t1 -chained-pch
// RUN: %clang_cc1 -fsyntax-only -verify -include-pch %t2 %s
// RUN: %clang_cc1 -ast-print -include-pch %t2 %s | FileCheck %s

// CHECK: void f();
// CHECK: void g();

void h() {
  f();
  g();
}
