// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-templates.h -verify %s -ast-dump

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -include-pch %t -verify %s -ast-dump 

struct A {
  typedef int type;
  static void my_f();
  template <typename T>
  static T my_templf(T x) { return x; }
};

void test() {
  int x = templ_f<int, 5>(3);
  
  S<char, float>::templ();
  S<int, char>::partial();
  S<int, float>::explicit_special();
  
  Dep<A>::Ty ty;
  Dep<A> a;
  a.f();
}
