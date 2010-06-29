// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-friends.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-friends.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

class F {
  void m() {
    A* a;
    a->x = 0;
  }
};
