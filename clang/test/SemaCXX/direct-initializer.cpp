// RUN: %clang_cc1 -fsyntax-only -verify %s 

int x(1);
int (x2)(1);

void f() {
  int x(1);
  int (x2)(1);
  for (int x(1);;) {}
}

class Y { 
  explicit Y(float);
};

class X { // expected-note{{candidate function}}
public:
  explicit X(int); // expected-note{{candidate function}}
  X(float, float, float); // expected-note{{candidate function}}
  X(float, Y); // expected-note{{candidate function}}
};

class Z {
public:
  Z(int);
};

void g() {
  X x1(5);
  X x2(1.0, 3, 4.2);
  X x3(1.0, 1.0); // expected-error{{no matching constructor for initialization of 'x3'; candidates are:}}
  Y y(1.0);
  X x4(3.14, y);

  Z z; // expected-error{{no matching constructor for initialization of 'z'}}
}

struct Base {
   operator int*() const; 
};

struct Derived : Base {
   operator int*(); 
};

void foo(const Derived cd, Derived d) {
        int *pi = cd;	// expected-error {{incompatible type initializing 'struct Derived const', expected 'int *'}}
        int *ppi = d; 

}
