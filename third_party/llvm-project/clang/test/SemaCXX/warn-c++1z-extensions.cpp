// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s

void f() {
  if (bool b = true; b) {} // expected-warning {{'if' initialization statements are a C++17 extension}}
  switch (int n = 5; n) { // expected-warning {{'switch' initialization statements are a C++17 extension}}
  case 5: break;
  }
}
