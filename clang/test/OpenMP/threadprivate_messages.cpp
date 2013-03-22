// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 %s

#pragma omp threadprivate // expected-error {{expected '(' after 'threadprivate'}}
#pragma omp threadprivate( // expected-error {{expected unqualified-id}}
#pragma omp threadprivate() // expected-error {{expected unqualified-id}}
#pragma omp threadprivate(1) // expected-error {{expected unqualified-id}}
struct CompleteSt{
 int a;
};

struct CompleteSt1{
#pragma omp threadprivate(1) // expected-error {{expected unqualified-id}}
 int a;
} d; // expected-note {{forward declaration of 'd'}}

int a; // expected-note {{forward declaration of 'a'}}

#pragma omp threadprivate(a)
#pragma omp threadprivate(u) // expected-error {{use of undeclared identifier 'u'}}
#pragma omp threadprivate(d, a) // expected-error {{'#pragma omp threadprivate' must precede all references to variable 'a'}}
int foo() { // expected-note {{declared here}}
  static int l;
#pragma omp threadprivate(l)) // expected-warning {{extra tokens at end of '#pragma omp threadprivate' are ignored}}
  return (a);
}

#pragma omp threadprivate a // expected-error {{expected '(' after 'threadprivate'}}
#pragma omp threadprivate(d // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{'#pragma omp threadprivate' must precede all references to variable 'd'}}
#pragma omp threadprivate(d))
int x, y;
#pragma omp threadprivate(x)) // expected-warning {{extra tokens at end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate(y)), // expected-warning {{extra tokens at end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate(a,d)  // expected-error {{'#pragma omp threadprivate' must precede all references to variable 'a'}} expected-error {{'#pragma omp threadprivate' must precede all references to variable 'd'}}
#pragma omp threadprivate(d.a) // expected-error {{expected unqualified-id}}
#pragma omp threadprivate((float)a) // expected-error {{expected unqualified-id}}
int foa;
#pragma omp threadprivate(faa) // expected-error {{use of undeclared identifier 'faa'; did you mean 'foa'?}}
#pragma omp threadprivate(foo) // expected-error {{'foo' is not a global variable, static local variable or static data member}}
#pragma omp threadprivate (int a=2) // expected-error {{expected unqualified-id}}

struct IncompleteSt; // expected-note {{forward declaration of 'IncompleteSt'}}

extern IncompleteSt e;
#pragma omp threadprivate (e) // expected-error {{a threadprivate variable must not have incomplete type 'IncompleteSt'}}

int &f = a; // expected-note {{forward declaration of 'f'}}
#pragma omp threadprivate (f) // expected-error {{arguments of '#pragma omp threadprivate' cannot be of reference type 'int &'}}

class Class {
  private:
    int a; // expected-note {{declared here}}
    static int b;
    Class() : a(0){}
  public:
    Class (int aaa) : a(aaa) {}
#pragma omp threadprivate (b, a) // expected-error {{'a' is not a global variable, static local variable or static data member}}
} g(10);
#pragma omp threadprivate (b) // expected-error {{use of undeclared identifier 'b'}}
#pragma omp threadprivate (Class::b) // expected-error {{expected unqualified-id}}
#pragma omp threadprivate (g)

namespace ns {
  int m;
#pragma omp threadprivate (m)
}
#pragma omp threadprivate (m) // expected-error {{use of undeclared identifier 'm'}}
#pragma omp threadprivate (ns::m) // expected-error {{expected unqualified-id}}
#pragma omp threadprivate (ns:m) // expected-error {{expected unqualified-id}}

const int h = 12;
const volatile int i = 10;
#pragma omp threadprivate (h, i)


template <class T>
class TempClass {
  private:
    T a;
    TempClass() : a(){}
  public:
    TempClass (T aaa) : a(aaa) {}
    static T s;
#pragma omp threadprivate (s)
};
#pragma omp threadprivate (s) // expected-error {{use of undeclared identifier 's'}}

static __thread int t; // expected-note {{forward declaration of 't'}}
#pragma omp threadprivate (t) // expected-error {{variable 't' cannot be threadprivate because it is thread-local}}

int o; // expected-note {{candidate found by name lookup is 'o'}}
namespace {
int o; // expected-note {{candidate found by name lookup is '<anonymous namespace>::o'}}
}
#pragma omp threadprivate (o) // expected-error {{reference to 'o' is ambiguous}}

int main(int argc, char **argv) { // expected-note {{forward declaration of 'argc'}}

  int x, y = argc; // expected-note {{forward declaration of 'y'}}
  static double d1;
  static double d2;
  static double d3; // expected-note {{forward declaration of 'd3'}}

  d.a = a;
  d2++;
  ;
#pragma omp threadprivate(argc+y) // expected-error {{expected unqualified-id}}
#pragma omp threadprivate(argc,y) // expected-error 2 {{arguments of '#pragma omp threadprivate' must have static storage duration}}
#pragma omp threadprivate(d2) // expected-error {{'#pragma omp threadprivate' must precede all references to variable 'd2'}}
#pragma omp threadprivate(d1)
  {
  ++a;d2=0;
#pragma omp threadprivate(d3) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'd3' variable declaration}}
  }
#pragma omp threadprivate(d3)

#pragma omp threadprivate(a) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'a' variable declaration}}
  return (y);
#pragma omp threadprivate(d) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'd' variable declaration}}
}
