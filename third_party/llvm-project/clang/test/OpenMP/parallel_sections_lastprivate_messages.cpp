// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-version=50 -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp-simd %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-version=50 -fopenmp-simd %s -Wuninitialized

extern int omp_default_mem_alloc;
void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  S2(S2 &s2) : a(s2.a) {}
  const S2 &operator=(const S2 &) const;
  static float S2s; // expected-note {{static data member is predetermined as shared}}
  static const float S2sc; // expected-note {{'S2sc' declared here}}
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
  S3 &operator=(const S3 &s3); // expected-note 2 {{implicitly declared private here}}

public:
  S3() : a(0) {}
  S3(S3 &s3) : a(s3.a) {}
};
const S3 c;         // expected-note {{'c' defined here}}
const S3 ca[5];     // expected-note {{'ca' defined here}}
extern const int f; // expected-note {{'f' declared here}}
class S4 {
  int a;
  S4();             // expected-note 3 {{implicitly declared private here}}
  S4(const S4 &s4);

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}}

public:
  S5(const S5 &s5) : a(s5.a) {}
  S5(int v) : a(v) {}
};
class S6 {
  int a;
  S6() : a(0) {}

public:
  S6(const S6 &s6) : a(s6.a) {}
  S6(int v) : a(v) {}
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C>
int foomain(int argc, char **argv) {
  I e(4); // omp50-note {{'e' defined here}}
  I g(5);
  int i;
  int &j = i;
#pragma omp parallel sections lastprivate // expected-error {{expected '(' after 'lastprivate'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate() // expected-error {{expected expression}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(conditional: e,argc) lastprivate(conditional: // omp50-error {{expected expression}} omp45-error 2 {{use of undeclared identifier 'conditional'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp50-error {{expected list item of scalar type in 'lastprivate' clause with 'conditional' modifier}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(foo:argc) // omp50-error {{expected 'conditional' in OpenMP clause 'lastprivate'}} omp45-error {{expected ',' or ')' in 'lastprivate' clause}} omp45-error {{expected ')'}} omp45-error {{expected variable name}} omp45-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(S1) // expected-error {{'S1' does not refer to a value}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(a, b) // expected-error {{lastprivate variable with incomplete type 'S1'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argv[1]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(e, g) // expected-error 2 {{calling a private constructor of class 'S4'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(h) // expected-error {{threadprivate or thread local variable cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections linear(i) // expected-error {{unexpected OpenMP clause 'linear' in directive '#pragma omp parallel sections'}}
  {
    foo();
  }
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp parallel sections lastprivate(i)
    {
      foo();
    }
    v += i;
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp parallel sections lastprivate(j)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(i)
  {
    foo();
  }
  return 0;
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5;       // expected-note {{'d' defined here}}
  const int da[5] = {0}; // expected-note {{'da' defined here}}
  S4 e(4);
  S5 g(5);
  S3 m;
  S6 n(2);
  int i;
  int &j = i;
#pragma omp parallel sections lastprivate // expected-error {{expected '(' after 'lastprivate'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate() // expected-error {{expected expression}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argc)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(S1) // expected-error {{'S1' does not refer to a value}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(a, b, c, d, f) // expected-error {{lastprivate variable with incomplete type 'S1'}} expected-error 1 {{const-qualified variable without mutable fields cannot be lastprivate}} expected-error 2 {{const-qualified variable cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(argv[1]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(2 * 2) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(ba)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(ca) // expected-error {{const-qualified variable without mutable fields cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(da) // expected-error {{const-qualified variable cannot be lastprivate}}
  {
    foo();
  }
  int xa;
#pragma omp parallel sections lastprivate(xa) // OK
  {
    foo();
  }
#pragma omp parallel sections lastprivate(S2::S2s) // expected-error {{shared variable cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(S2::S2sc) // expected-error {{const-qualified variable cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections safelen(5) // expected-error {{unexpected OpenMP clause 'safelen' in directive '#pragma omp parallel sections'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(m) // expected-error {{'operator=' is a private member of 'S3'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be lastprivate}}
  {
    foo();
  }
#pragma omp parallel sections private(xa), lastprivate(xa) // expected-error {{private variable cannot be lastprivate}} expected-note {{defined as private}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(i)
  {
    foo();
  }
#pragma omp parallel private(xa)
#pragma omp parallel sections lastprivate(xa)
  {
    foo();
  }
#pragma omp parallel reduction(+ : xa)
#pragma omp parallel sections lastprivate(xa)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(j)
  {
    foo();
  }
#pragma omp parallel sections firstprivate(m) lastprivate(m) // expected-error {{'operator=' is a private member of 'S3'}}
  {
    foo();
  }
#pragma omp parallel sections lastprivate(n) firstprivate(n) // OK
  {
    foo();
  }
  static int r;
#pragma omp parallel sections lastprivate(r) // OK
  {
    foo();
  }
  return foomain<S4, S5>(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<S4, S5>' requested here}}
}
