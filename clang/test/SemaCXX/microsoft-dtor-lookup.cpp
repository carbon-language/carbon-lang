// RUN: %clang_cc1 -triple i686-pc-win32 -cxx-abi itanium -fsyntax-only %s
// RUN: %clang_cc1 -triple i686-pc-win32 -cxx-abi microsoft -verify -DMSVC_ABI %s

namespace Test1 {

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

}

namespace Test2 {

// In the MSVC ABI, functions must destroy their aggregate arguments.  foo
// requires a dtor for B, but we can't implicitly define it because ~A is
// private.  bar should be able to call A's private dtor without error, even
// though MSVC rejects bar.

class A {
private:
  ~A(); // expected-note 2{{declared private here}}
  int a;
};

struct B : public A { // expected-error {{base class 'Test2::A' has private destructor}}
  int b;
};

struct C {
  ~C();
  int c;
};

struct D {
  // D has a non-trivial implicit dtor that destroys C.
  C o;
};

void foo(B b) { } // expected-note {{implicit destructor for 'Test2::B' first required here}}
void bar(A a) { } // expected-error {{variable of type 'Test2::A' has private destructor}}
void baz(D d) { } // no error

}

#ifdef MSVC_ABI
namespace Test3 {

class A {
  A();
  ~A(); // expected-note 2{{implicitly declared private here}}
  friend void bar(A);
  int a;
};

void bar(A a) { }
void baz(A a) { } // expected-error {{variable of type 'Test3::A' has private destructor}}

// MSVC accepts foo() but we reject it for consistency with Itanium.  MSVC also
// rejects this if A has a copy ctor or if we call A's ctor.
void foo(A *a) {
  bar(*a); // expected-error {{temporary of type 'Test3::A' has private destructor}}
}
}
#endif

namespace Test4 {
// Don't try to access the dtor of an incomplete on a function declaration.
class A;
void foo(A a);
}
