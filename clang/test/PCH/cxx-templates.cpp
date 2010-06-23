// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-templates.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

void test() {
  int x = templ_f(3);
  
  S<char, float>::templ();
  S<int, char>::partial();
  S<int, float>::explicit_special();
}
