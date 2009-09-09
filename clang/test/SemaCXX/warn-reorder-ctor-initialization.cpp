// RUN: clang-cc  -fsyntax-only -Wreorder -verify %s

struct BB {};

struct BB1 {};

class complex : public BB, BB1 { 
public: 
  complex() : s2(1),  // expected-warning {{member 's2' will be initialized after}}
              s1(1) , // expected-note {{field s1}} 
              s3(3),  // expected-warning {{member 's3' will be initialized after}} 
              BB1(),   // expected-note {{base 'struct BB1'}}  \
                      // expected-warning {{base class 'struct BB1' will be initialized after}}
              BB() {}  // expected-note {{base 'struct BB'}}
  int s1;
  int s2;
  int s3;
}; 


// testing virtual bases.


struct V { 
  V();
};

struct A : public virtual V { 
  A(); 
};

struct B : public virtual V {
  B(); 
};

struct Diamond : public A, public B {
  Diamond() : A(), B() {}
};


struct C : public A, public B, private virtual V { 
  C() { }
};


struct D : public A, public B { 
  D()  : A(), V() {   } // expected-warning {{base class 'struct A' will be initialized after}} \
                        // expected-note {{base 'struct V'}}
};


struct E : public A, public B, private virtual V { 
  E()  : A(), V() {  } // expected-warning {{base class 'struct A' will be initialized after}} \
                       // expected-note {{base 'struct V'}}
};


struct A1  { 
  A1(); 
};

struct B1 {
  B1();
};

struct F : public A1, public B1, private virtual V { 
  F()  : A1(), V() {  } // expected-warning {{base class 'struct A1' will be initialized after}} \
                        // expected-note {{base 'struct V'}}
};

struct X : public virtual A, virtual V, public virtual B {
  X(): A(), V(), B() {} // expected-warning {{base class 'struct A' will be initialized after}} \
                        // expected-note {{base 'struct V'}}
};

class Anon {
  int c; union {int a,b;}; int d;
  Anon() : c(10), b(1), d(2) {}
};
class Anon2 {
  int c; union {int a,b;}; int d;
  Anon2() : c(2),
            d(10), // expected-warning {{member 'd' will be initialized after}}
            b(1) {} // expected-note {{field b}}
};
class Anon3 {
  union {int a,b;};
  Anon3() : b(1) {}
};
