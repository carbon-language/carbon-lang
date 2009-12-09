// RUN: clang-cc -fsyntax-only -verify %s
int g(int);

void f() {
  int i;
  int &r = i;
  r = 1;
  int *p = &r;
  int &rr = r;
  int (&rg)(int) = g;
  rg(i);
  int a[3];
  int (&ra)[3] = a;
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
    int (&rc)[3] = c;
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
  double& rd2 = 2.0; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}
  int i = 2;
  double& rd3 = i; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}

  const A& rca = fB();
}

void test5() {
  //  const double& rcd2 = 2; // rcd2 refers to temporary with value 2.0
  const volatile int cvi = 1;
  const int& r = cvi; // expected-error{{binding of reference to type 'int const' to a value of type 'int const volatile' drops qualifiers}}
}

// C++ [dcl.init.ref]p3
int& test6(int& x) {
  int& yo; // expected-error{{declaration of reference variable 'yo' requires an initializer}}

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

// C++ [dcl.ref]p1, C++ [dcl.ref]p4
void test8(int& const,// expected-error{{'const' qualifier may not be applied to a reference}}
           
           void&,     // expected-error{{cannot form a reference to 'void'}}
           int& &)    // expected-error{{type name declared as a reference to a reference}}
{
  typedef int& intref;
  typedef intref& intrefref; // C++ DR 106: reference collapsing

  typedef intref const intref_c; // okay. FIXME: how do we verify that this is the same type as intref?
}


class string {
  char *Data;
  unsigned Length;
public:
  string(); 
  ~string();
};

string getInput();

void test9() {
  string &s = getInput(); // expected-error{{lvalue reference}}
}
