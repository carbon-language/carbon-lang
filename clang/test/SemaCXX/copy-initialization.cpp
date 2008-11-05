// RUN: clang -fsyntax-only -verify %s 

class X {
public:
  explicit X(const X&);
  X(int*); // expected-note{{candidate function}}
  explicit X(float*);
};

class Y : public X { };

void f(Y y, int *ip, float *fp) {
  X x1 = y; // expected-error{{no matching constructor for initialization of 'x1'; candidates are:}}
  X x2 = 0;
  X x3 = ip;
  X x4 = fp; // expected-error{{incompatible type initializing 'x4', expected 'class X'}}
}
