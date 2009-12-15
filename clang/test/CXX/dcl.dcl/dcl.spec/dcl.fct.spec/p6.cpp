// RUN: %clang_cc1 -verify %s
// XFAIL: *

class A {
public:
  explicit A();
  
  explicit operator int(); // expected-warning {{explicit conversion functions are a C++0x extension}}

  explicit void f0(); // expected-error {{'explicit' cannot only be applied to constructor or conversion function}}
};

explicit A::A() { } // expected-error {{'explicit' cannot be specified outside class definition}}
