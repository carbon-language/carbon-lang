// RUN: clang-cc -fsyntax-only %s -Wnon-pod-varargs
class X { };

int& copycon(X x);
float& copycon(...);

void test_copycon(X x, X const xc, X volatile xv) {
  int& i1 = copycon(x);
  int& i2 = copycon(xc);
  float& f1 = copycon(xv);
}

class A {
public:
  A(A&);
};

class B : public A { };

short& copycon2(A a);
int& copycon2(B b);
float& copycon2(...);

void test_copycon2(A a, const A ac, B b, B const bc, B volatile bv) {
  int& i1 = copycon2(b);
  float& f1 = copycon2(bc); // expected-warning {{cannot pass object of non-POD type}}
  float& f2 = copycon2(bv); // expected-warning {{cannot pass object of non-POD type}}
  short& s1 = copycon2(a);
  float& f3 = copycon2(ac); // expected-warning {{cannot pass object of non-POD type}}
}

int& copycon3(A a);
float& copycon3(...);

void test_copycon3(B b, const B bc) {
  int& i1 = copycon3(b);
  float& f1 = copycon3(bc); // expected-warning {{cannot pass object of non-POD type}}
}


class C : public B { };

float& copycon4(A a);
int& copycon4(B b);

void test_copycon4(C c) {
  int& i = copycon4(c);
};
