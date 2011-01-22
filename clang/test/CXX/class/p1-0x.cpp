// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
namespace Test1 {

class A final { };
class B explicit { };
class C final explicit { };
class D final final { }; // expected-error {{class already marked 'final'}}
class E explicit explicit { }; // expected-error {{class already marked 'explicit'}}

}
