// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
namespace Test1 {

class A final { }; // expected-note {{'A' declared here}}
class B : A { }; // expected-error {{base 'A' is marked 'final'}}

}

