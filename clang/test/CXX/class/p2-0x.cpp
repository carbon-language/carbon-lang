// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
namespace Test1 {

class A final { }; // expected-note {{'A' declared here}}
class B : A { }; // expected-error {{base 'A' is marked 'final'}}

}

namespace Test2 {

template<typename T> struct A final { }; // expected-note 2 {{'A' declared here}}
struct B : A<int> { }; // expected-error {{base 'A' is marked 'final'}}
  
template<typename T> struct C : A<T> { }; // expected-error {{base 'A' is marked 'final'}}
struct D : C<int> { }; // expected-note {{in instantiation of template class 'Test2::C<int>' requested here}}

}

namespace Test3 {

template<typename T> struct A { };
template<> struct A<int> final { }; // expected-note {{'A' declared here}}

struct B : A<bool> { };
struct C : A<int> { }; // expected-error {{base 'A' is marked 'final'}}

}

