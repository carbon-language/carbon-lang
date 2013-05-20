// RUN: %clang_cc1 -triple i686-pc-win32 -cxx-abi itanium -fsyntax-only %s
// RUN: %clang_cc1 -triple i686-pc-win32 -cxx-abi microsoft -verify %s

// Should be accepted under the Itanium ABI (first RUN line) but rejected
// under the Microsoft ABI (second RUN line), as Microsoft ABI requires
// operator delete() lookups to be done at all virtual destructor declaration
// points.

struct A {
  void operator delete(void *); // expected-note {{member found by ambiguous name lookup}}
};

struct B {
  void operator delete(void *); // expected-note {{member found by ambiguous name lookup}}
};

struct C : A, B {
  ~C();
};

struct VC : A, B {
  virtual ~VC(); // expected-error {{member 'operator delete' found in multiple base classes of different types}}
};
