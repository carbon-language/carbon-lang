// Test C++ chained PCH functionality

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %S/Inputs/chain-cxx1.h -include %S/Inputs/chain-cxx2.h %s

// With PCH
// RUN: %clang_cc1 -x c++ -emit-pch -o %t1 %S/Inputs/chain-cxx1.h
// RUN: %clang_cc1 -x c++ -emit-pch -o %t2 %S/Inputs/chain-cxx2.h -include-pch %t1 -chained-pch
// RUN: %clang_cc1 -fsyntax-only -verify -include-pch %t2 %s

void test() {
  f();
  f(1);
  pf();
  f2();

  ns::g();
  ns::g(1);
  ns::pg();
  ns::g2();

  //typedef S<int>::I J;
}
