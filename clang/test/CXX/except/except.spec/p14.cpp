// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -verify -std=c++0x %s
struct A { };
struct B { };
struct C { };

// Destructor
struct X0 { 
  virtual ~X0() throw(A); // expected-note{{overridden virtual function is here}} 
};
struct X1 { 
  virtual ~X1() throw(B); // expected-note{{overridden virtual function is here}} 
};
struct X2 : public X0, public X1 { }; // expected-error 2{{exception specification of overriding function is more lax than base version}}
 
// Copy-assignment operator.
struct CA0 {
  CA0 &operator=(const CA0&) throw(A);
};
struct CA1 {
  CA1 &operator=(const CA1&) throw(B);
};
struct CA2 : CA0, CA1 { };

void test_CA() {
  CA2 &(CA2::*captr1)(const CA2&) throw(A, B) = &CA2::operator=;
  CA2 &(CA2::*captr2)(const CA2&) throw(A, B, C) = &CA2::operator=;
  CA2 &(CA2::*captr3)(const CA2&) throw(A) = &CA2::operator=; // expected-error{{target exception specification is not superset of source}}
  CA2 &(CA2::*captr4)(const CA2&) throw(B) = &CA2::operator=; // expected-error{{target exception specification is not superset of source}}
}

// In-class member initializers.
struct IC0 {
  int inClassInit = 0;
};
struct IC1 {
  int inClassInit = (throw B(), 0);
};
// FIXME: the exception specification on the default constructor is wrong:
// we cannot currently compute the set of thrown types.
static_assert(noexcept(IC0()), "IC0() does not throw");
static_assert(!noexcept(IC1()), "IC1() throws");
