// RUN: %clang_cc1 -fsyntax-only -verify %s

class Base { // expected-error {{cannot define the implicit default assignment operator for 'Base', because non-static reference member 'ref' can't use default assignment operator}} \
  // expected-warning{{class 'Base' does not declare any constructor to initialize its non-modifiable members}} \
  // expected-note {{synthesized method is first required here}}
  int &ref;  // expected-note {{declared here}} \
  // expected-note{{reference member 'ref' will never be initialized}}
};

class X  : Base {  // // expected-error {{cannot define the implicit default assignment operator for 'X', because non-static const member 'cint' can't use default assignment operator}} \
  // expected-note  {{synthesized method is first required here}}
public:
  X();
  const int cint;  // expected-note {{declared here}}
}; 

struct Y  : X { 
  Y();
  Y& operator=(const Y&);
  Y& operator=(volatile Y&);
  Y& operator=(const volatile Y&);
  Y& operator=(Y&);
}; 

class Z : Y {};

Z z1;
Z z2;

// Test1
void f(X x, const X cx) {
  x = cx;
  x = cx;
  z1 = z2;
}

// Test2
class T {};
T t1;
T t2;

void g() {
  t1 = t2;
}

// Test3
class V {
public:
  V();
  V &operator = (V &b);
};

class W : V {};
W w1, w2;

void h() {
  w1 = w2;
}

// Test4

class B1 {
public:
  B1();
  B1 &operator = (B1 b);
};

class D1 : B1 {};
D1 d1, d2;

void i() {
  d1 = d2;
}

// Test5

class E1 { // expected-error{{cannot define the implicit default assignment operator for 'E1', because non-static const member 'a' can't use default assignment operator}} \
  // expected-note {{synthesized method is first required here}}

public:
  const int a; // expected-note{{declared here}}
  E1() : a(0) {}  

};

E1 e1, e2;

void j() {
  e1 = e2; 
}

