// RUN: clang -fsyntax-only %s 
class Z { };

class Y { 
public:
  Y(const Z&);
};

class X {
public:
  X(int);
  X(const Y&);
};

void f(X);

void g(short s, Y y, Z z) {
  f(s);
  f(1.0f);
  f(y);
  f(z); // expected-error{{incompatible}}
}

