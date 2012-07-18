// Test this without pch.
// RUN: %clang -include %S/cxx-typeid.h -fsyntax-only -std=c++11 -Xclang -verify %s

// RUN: %clang -ccc-pch-is-pch -std=c++11 -x c++-header -o %t.gch %S/cxx-typeid.h
// RUN: %clang -ccc-pch-is-pch -std=c++11 -include %t -fsyntax-only -Xclang -verify %s

void f() {
    (void)typeid(int);
}
