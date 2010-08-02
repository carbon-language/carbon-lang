// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-typeid.h -fsyntax-only -verify %s

// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-typeid.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

void f() {
    (void)typeid(int);
}
