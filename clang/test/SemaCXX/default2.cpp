// RUN: clang-cc -fsyntax-only -verify %s

void f(int i, int j, int k = 3);
void f(int i, int j, int k);
void f(int i, int j = 2, int k);
void f(int i, int j, int k);
void f(int i = 1, int j, int k);
void f(int i, int j, int k);

void i()
{
  f();
  f(0);
  f(0, 1);
  f(0, 1, 2);
}


int f1(int i, int i, int j) { // expected-error {{redefinition of parameter 'i'}}
  i = 17;
  return j;
} 

int x;
void g(int x, int y = x); // expected-error {{default argument references parameter 'x'}}

void h()
{
   int i;
   extern void h2(int x = sizeof(i)); // expected-error {{default argument references local variable 'i' of enclosing function}}
}

void g2(int x, int y, int z = x + y); // expected-error {{default argument references parameter 'x'}} expected-error {{default argument references parameter 'y'}}

void nondecl(int (*f)(int x = 5)) // {expected-error {{default arguments can only be specified}}}
{
  void (*f2)(int = 17)  // {expected-error {{default arguments can only be specified}}}
    = (void (*)(int = 42))f; // {expected-error {{default arguments can only be specified}}}
}

class X {
  void f(X* x = this); // expected-error{{invalid use of 'this' outside of a nonstatic member function}}

  void g() { 
    int f(X* x = this); // expected-error{{default argument references 'this'}}
  }
};

// C++ [dcl.fct.default]p6
class C { 
  static int x;
  void f(int i = 3); // expected-note{{previous definition is here}}
  void g(int i, int j = x); 

  void h();
}; 
void C::f(int i = 3) // expected-error{{redefinition of default argument}}
{ } 

void C::g(int i = 88, int j) {}

void C::h() {
  g(); // okay
}

// C++ [dcl.fct.default]p9
struct Y { 
  int a; 
  int mem1(int i = a); // expected-error{{invalid use of nonstatic data member 'a'}}
  int mem2(int i = b); // OK; use Y::b 
  int mem3(int i);
  int mem4(int i);

  struct Nested {
    int mem5(int i = b, // OK; use Y::b
             int j = c, // OK; use Y::Nested::c
             int k = j, // expected-error{{default argument references parameter 'j'}}
             int l = a,  // expected-error{{invalid use of nonstatic data member 'a'}}
             Nested* self = this, // expected-error{{invalid use of 'this' outside of a nonstatic member function}}
             int m); // expected-error{{missing default argument on parameter 'm'}}
    static int c;
  };

  static int b; 

  int (*f)(int = 17); // expected-error{{default arguments can only be specified for parameters in a function declaration}}

  void mem8(int (*fp)(int) = (int (*)(int = 17))0); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
}; 

int Y::mem3(int i = b) { return i; } // OK; use X::b

int Y::mem4(int i = a) // expected-error{{invalid use of nonstatic data member 'a'}}
{ return i; }


// Try to verify that default arguments interact properly with copy
// constructors.
class Z {
public:
  Z(Z&, int i = 17); // expected-note 2 {{candidate function}}

  void f(Z& z) { 
    Z z2;    // expected-error{{no matching constructor for initialization}}
    Z z3(z);
  }

  void test_Z(const Z& z) {
    Z z2(z); // expected-error{{no matching constructor for initialization of 'z2'}}
  }
};

void test_Z(const Z& z) {
  Z z2(z); // expected-error{{no matching constructor for initialization of 'z2'}}
}

struct ZZ {
  void f(ZZ z = g()); // expected-error{{no matching constructor for initialization}}

  static ZZ g(int = 17);

  ZZ(ZZ&, int = 17); // expected-note{{candidate function}}
};
