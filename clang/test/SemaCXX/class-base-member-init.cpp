// RUN: clang-cc -fsyntax-only -verify %s

class S {
public:
  S (); 
};

struct D : S {
  D() : b1(0), b2(1), b1(0), S(), S() {} // expected-error {{multiple initializations given for non-static member 'b1'}} \
					 // expected-error {{multiple initializations given for base 'class S'}}

  int b1;
  int b2;

};


