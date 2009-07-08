// RUN: clang-cc -fsyntax-only -verify %t
class A { 
public:
  int& i; 

  A(int& i) : i(i) { }
  
  static int s;
}; 

void f() {
  int b;
  A a(b); 
  
  int A::*ip = &A::s; // expected-error {{incompatible type initializing 'int *', expected 'int class A::*'}}
  a.*&A::s = 10; // expected-error{{right hand operand to .* has non pointer-to-member type 'int *'}}
  a.*&A::i = 10; // expected-error{{cannot form a pointer-to-member to member 'i' of reference type 'int &'}}

  void A::*p = 0; // expected-error{{'p' declared as a member pointer to void}}
}
