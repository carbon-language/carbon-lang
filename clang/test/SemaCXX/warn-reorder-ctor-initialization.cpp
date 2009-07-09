// RUN: clang-cc  -fsyntax-only -Wreorder -verify %s

struct B {};

struct B1 {};

class complex : public B, B1 { 
public: 
  complex() : s2(1),  // expected-warning {{member 's2' will be initialized after}}
              s1(1) , // expected-note {{field s1}} 
              s3(3),  // expected-warning {{member 's3' will be initialized after}} 
              B1(),   // expected-note {{base 'struct B1'}}  \
                      // expected-warning {{base class 'struct B1' will be initialized after}}
              B() {}  // expected-note {{base 'struct B'}}
  int s1;
  int s2;
  int s3;
}; 
