// RUN: %clang_cc1 -cxx-abi microsoft -fms-compatibility -fsyntax-only -verify %s

// Test that we don't allow reinterpret_casts from pointers of one size to
// pointers of a different size.
struct A {};
struct B {};
struct C: A, B {};

void (A::*paf)();
void (C::*pcf)() = reinterpret_cast<void (C::*)()>(paf); //expected-error{{cannot reinterpret_cast from member pointer type}}

class __single_inheritance D; 
class __multiple_inheritance D; // expected-warning {{ignored since inheritance model was already declared as 'single'}}
  
class __virtual_inheritance E;
class __virtual_inheritance E;  // no warning expected since same attribute
