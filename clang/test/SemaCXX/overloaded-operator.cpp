// RUN: clang -fsyntax-only -verify %s 
class X { };

X operator+(X, X);

void f(X x) {
  x = x + x;
}

struct Y;
struct Z;

struct Y {
  Y(const Z&);
};

struct Z {
  Z(const Y&);
};

Y operator+(Y, Y);
bool operator-(Y, Y); // expected-note{{candidate function}}
bool operator-(Z, Z); // expected-note{{candidate function}}

void g(Y y, Z z) {
  y = y + z;
  bool b = y - z; // expected-error{{use of overloaded operator '-' is ambiguous; candidates are:}}
}

struct A {
  bool operator==(Z&); // expected-note{{candidate function}}
};

A make_A();

bool operator==(A&, Z&); // expected-note{{candidate function}}

void h(A a, const A ac, Z z) {
  make_A() == z;
  a == z; // expected-error{{use of overloaded operator '==' is ambiguous; candidates are:}}
  ac == z; // expected-error{{invalid operands to binary expression ('struct A const' and 'struct Z')}}
}

struct B {
  bool operator==(const B&) const;

  void test(Z z) {
    make_A() == z;
  }
};
