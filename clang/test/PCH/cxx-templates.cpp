// Test this without pch.
// RUN: %clang_cc1 -fexceptions -include %S/cxx-templates.h -verify %s -ast-dump -o -
// RUN: %clang_cc1 -fexceptions -include %S/cxx-templates.h %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -fexceptions -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -fexceptions -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -fexceptions -include-pch %t %s -emit-llvm -o - | FileCheck %s

// CHECK: define weak_odr void @_ZN2S4IiE1mEv
// CHECK: define linkonce_odr void @_ZN2S3IiE1mEv

struct A {
  typedef int type;
  static void my_f();
  template <typename T>
  static T my_templf(T x) { return x; }
};

void test(const int (&a6)[17]) {
  int x = templ_f<int, 5>(3);
  
  S<char, float>::templ();
  S<int, char>::partial();
  S<int, float>::explicit_special();
  
  Dep<A>::Ty ty;
  Dep<A> a;
  a.f();
  
  S3<int> s3;
  s3.m();

  TS5 ts(0);

  S6<const int[17]>::t2 b6 = a6;
}

template struct S4<int>;

S7<int[5]> s7_5;

namespace ZeroLengthExplicitTemplateArgs {
  template void f<X>(X*);
}
