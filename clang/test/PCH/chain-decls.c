// Test this without pch.
// RUN: %clang_cc1 -include %S/Inputs/chain-decls1.h -include %S/Inputs/chain-decls2.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t1 %S/Inputs/chain-decls1.h
// RUN: %clang_cc1 -emit-pch -o %t2 %S/Inputs/chain-decls2.h -include-pch %t1 -chained-pch
// RUN: %clang_cc1 -include-pch %t2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -ast-print -include-pch %t2 %s | FileCheck %s

// CHECK: void f();
// CHECK: void g();

void h() {
  f();
  g();

  struct one x;
  one();
  struct two y;
  two();
  struct three z;
}
