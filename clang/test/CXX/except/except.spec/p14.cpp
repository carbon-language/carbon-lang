// RUN: %clang_cc1 -fexceptions -verify %s
struct A { };
struct B { };

struct X0 { 
  virtual ~X0() throw(A); // expected-note{{overridden virtual function is here}} 
};
struct X1 { 
  virtual ~X1() throw(B); // expected-note{{overridden virtual function is here}} 
};
struct X2 : public X0, public X1 { }; // expected-error 2{{exception specification of overriding function is more lax than base version}}
 
