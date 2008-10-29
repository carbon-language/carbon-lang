// RUN: clang -fsyntax-only -verify %s
int g(int);

void f() {
  int i;
  int &r = i;
  r = 1;
  int *p = &r;
  int &rr = r;
  int (&rg)(int) = g; // expected-warning{{statement was disambiguated as declaration}}
  rg(i);
  int a[3];
  int (&ra)[3] = a;  // expected-warning{{statement was disambiguated as declaration}}
  ra[1] = i;
  int *Q;
  int *& P = Q;
  P[1] = 1;
}

typedef int t[1];
void test2() {
    t a;
    t& b = a;


    int c[3];
    int (&rc)[3] = c; // expected-warning{{statement was disambiguated as declaration}}
}

// C++ [dcl.init.ref]p5b1
struct A { };
struct B : A { } b;

void test3() {
  double d = 2.0;
  double& rd = d; // rd refers to d
  const double& rcd = d; // rcd refers to d

  A& ra = b; // ra refers to A subobject in b
  const A& rca = b; // rca refers to A subobject in b
}

B fB();

// C++ [dcl.init.ref]p5b2
void test4() {
  double& rd2 = 2.0; // expected-error{{non-const reference to type 'double' cannot be initialized with a temporary of type 'double'}}
  int i = 2;
  double& rd3 = i; // expected-error{{non-const reference to type 'double' cannot be initialized with a value of type 'int'}}

  const A& rca = fB();
}

void test5() {
  const double& rcd2 = 2; // rcd2 refers to temporary with value 2.0
  const volatile int cvi = 1;
  const int& r = cvi; // expected-error{{initialization of reference to type 'int const' with a value of type 'int const volatile' drops qualifiers}}
}

// C++ [dcl.init.ref]p3
int& test6(int& x) {
  int& yo; // expected-error{{declaration of reference variable 'yo' requires an initializer}}


  const int val; // expected-error{{declaration of const variable 'val' requires an initializer}}

  return x;
}
int& not_initialized_error; // expected-error{{declaration of reference variable 'not_initialized_error' requires an initializer}}
extern int& not_initialized_okay;

class Test6 {
  int& okay;
};

struct C : B, A { };

void test7(C& c) {
  A& a1 = c; // expected-error {{ambiguous conversion from derived class 'struct C' to base class 'struct A':}}
}
