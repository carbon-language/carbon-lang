// Test this without pch.
// RUN: %clang_cc1 -include %S/Inputs/chain-macro-override1.h -include %S/Inputs/chain-macro-override2.h -fsyntax-only -verify -detailed-preprocessing-record %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t1 %S/Inputs/chain-macro-override1.h -detailed-preprocessing-record 
// RUN: %clang_cc1 -emit-pch -o %t2 %S/Inputs/chain-macro-override2.h -include-pch %t1 -chained-pch -detailed-preprocessing-record 
// RUN: %clang_cc1 -include-pch %t2 -fsyntax-only -verify %s

int foo() {
  f();
  g();
  h();
  return x;
}
