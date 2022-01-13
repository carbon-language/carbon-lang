// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 %s -Wuninitialized

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_null_allocator;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

void foo() {
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}} expected-note 1 {{forward declaration of 'S1'}} expected-note {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  static float S2s; // expected-note {{static data member is predetermined as shared}} expected-note 1 {{static data member is predetermined as shared}}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;

public:
  S3() : a(0) {}
};
const S3 c; // expected-note 2 {{'c' defined here}}
const S3 ca[5]; // expected-note 2 {{'ca' defined here}}
extern const int f; // expected-note 2 {{'f' declared here}}

int threadvar;
#pragma omp threadprivate(threadvar) // expected-note {{defined as threadprivate or thread local}} expected-note 1 {{defined as threadprivate or thread local}}

class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}} expected-note 1 {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}} expected-note 1 {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
};
namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}} expected-note 1 {{defined as threadprivate or thread local}} expected-note 2 {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C, class D, class E>
int foomain(I argc, C **argv) {
  const I d = 5; // expected-note {{'d' defined here}}
  const I da[5] = { 0 }; // expected-note {{'da' defined here}}
  D e(4);
  E g[] = {5, 6};
  I i;
  I &j = i;
#pragma omp target parallel private // expected-error {{expected '(' after 'private'}}
{}
#pragma omp target parallel private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private() // expected-error {{expected expression}}
{}
#pragma omp target parallel private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
{}
#pragma omp target parallel private(argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
{}
#pragma omp target parallel private(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
{}
#pragma omp target parallel private(S1) // expected-error {{'S1' does not refer to a value}}
{}
#pragma omp target parallel private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
{}
#pragma omp target parallel private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 1 {{const-qualified variable without mutable fields cannot be private}} expected-error 2 {{const-qualified variable cannot be private}}
{}
#pragma omp target parallel private(argv[1]) // expected-error {{expected variable name}}
{}
#pragma omp target parallel allocate(omp_thread_mem_alloc: ba) private(ba) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'target parallel' directive}} expected-error {{allocator must be specified in the 'uses_allocators' clause}}
{}
#pragma omp target parallel private(ca) // expected-error {{const-qualified variable without mutable fields cannot be private}}
{}
#pragma omp target parallel private(da) // expected-error {{const-qualified variable cannot be private}}
{}
#pragma omp target parallel private(S2::S2s) // expected-error {{shared variable cannot be private}}
{}
#pragma omp target parallel private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
{}
  #pragma omp target parallel private(threadvar, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be private}}
  {}
  #pragma omp target parallel shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
  #pragma omp target parallel firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp target parallel private(i)
  {}
  #pragma omp target parallel private(j)
  foo();
  #pragma omp parallel firstprivate(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp target parallel private(i)
    foo();
  }
  static int m;
  #pragma omp target parallel private(m) // OK
  foo();
#pragma omp target parallel private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
{}
#pragma omp target parallel private(B::x) // expected-error {{threadprivate or thread local variable cannot be private}}
#pragma omp parallel
  {
    int v = 0;
    int i;
  }
#pragma omp target parallel shared(i)
{}
#pragma omp target parallel private(i)
{}
#pragma omp target parallel private(j)
{}
#pragma omp target parallel private(i)
{}
  static int si;
#pragma omp target parallel private(si) // OK
  {}
  return 0;
}

void bar(S4 a[2]) {
#pragma omp parallel
#pragma omp target parallel private(a)
  {}
}

int main(int argc, char **argv) {
  const int d = 5; // expected-note {{'d' defined here}}
  const int da[5] = { 0 }; // expected-note {{'da' defined here}}
  S4 e(4);
  S5 g[] = {5, 6};
  int i;
  int &j = i;
#pragma omp target parallel private // expected-error {{expected '(' after 'private'}}
{}
#pragma omp target parallel private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private() // expected-error {{expected expression}}
{}
#pragma omp target parallel private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target parallel private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
{}
#pragma omp target parallel private(argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
{}
#pragma omp target parallel private(argc)
{}
#pragma omp target parallel private(S1) // expected-error {{'S1' does not refer to a value}}
{}
#pragma omp target parallel private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
{}
#pragma omp target parallel private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 1 {{const-qualified variable without mutable fields cannot be private}} expected-error 2 {{const-qualified variable cannot be private}}
{}
#pragma omp target parallel private(argv[1]) // expected-error {{expected variable name}}
{}
#pragma omp target parallel private(ba)
{}
#pragma omp target parallel private(ca) // expected-error {{const-qualified variable without mutable fields cannot be private}}
{}
#pragma omp target parallel private(da) // expected-error {{const-qualified variable cannot be private}}
{}
#pragma omp target parallel private(S2::S2s) // expected-error {{shared variable cannot be private}}
{}
#pragma omp target parallel private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
{}
  #pragma omp target parallel private(threadvar, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be private}}
  {}
  #pragma omp target parallel shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
  #pragma omp target parallel firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp target parallel private(i)
  {}
  #pragma omp target parallel private(j)
  foo();
  #pragma omp parallel firstprivate(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp target parallel private(i)
    foo();
  }
  static int m;
  #pragma omp target parallel private(m) // OK
  foo();
#pragma omp target parallel private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
{}
#pragma omp target parallel private(B::x) // expected-error {{threadprivate or thread local variable cannot be private}}
#pragma omp parallel
  {
    int i;
  }
#pragma omp target parallel shared(i)
{}
#pragma omp target parallel private(i)
{}
#pragma omp target parallel private(j)
{}
#pragma omp target parallel private(i)
{}
  static int si;
#pragma omp target parallel private(si) // OK
  {}
  return foomain<int, char, S4, S5>(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char, S4, S5>' requested here}}
}

