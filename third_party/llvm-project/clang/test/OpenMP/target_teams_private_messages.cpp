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

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} expected-note{{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  static float S2s; // expected-note {{static data member is predetermined as shared}}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 c; // expected-note {{'c' defined here}}
const S3 ca[5]; // expected-note {{'ca' defined here}}
extern const int f; // expected-note {{'f' declared here}}
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {} // expected-note {{implicitly declared private here}}
public:
  S5(int v):a(v) { }
};

int threadvar;
#pragma omp threadprivate(threadvar) // expected-note {{defined as threadprivate or thread local}}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5; // expected-note {{'d' defined here}}
  const int da[5] = { 0 }; // expected-note {{'da' defined here}}
  S4 e(4);
  S5 g(5);
  int i, z;
  int &j = i;
#pragma omp target teams private // expected-error {{expected '(' after 'private'}}
  foo();
#pragma omp target teams private ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams private () // expected-error {{expected expression}}
  foo();
#pragma omp target teams private (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams private (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams private (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
#pragma omp target teams private (argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
  foo();
#pragma omp target teams private (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma omp target teams private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 1 {{const-qualified variable without mutable fields cannot be private}} expected-error 2 {{const-qualified variable cannot be private}}
  foo();
#pragma omp target teams private (argv[1]) // expected-error {{expected variable name}}
  foo();
#pragma omp target teams private(ba) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  foo();
#pragma omp target teams private(ca) // expected-error {{const-qualified variable without mutable fields cannot be private}}
  foo();
#pragma omp target teams private(da) // expected-error {{const-qualified variable cannot be private}}
  foo();
#pragma omp target teams private(S2::S2s) // expected-error {{shared variable cannot be private}}
  foo();
#pragma omp target teams private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  foo();
#pragma omp target teams private(threadvar, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be private}}
  foo();
#pragma omp target teams shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
#pragma omp target teams firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
#pragma omp target teams private(i, z)
  foo();
#pragma omp target teams private(j)
  foo();
#pragma omp target teams firstprivate(i) uses_allocators(omp_thread_mem_alloc) allocate(omp_thread_mem_alloc: i) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'target teams' directive}}
  for (int k = 0; k < 10; ++k) {
#pragma omp parallel private(i)
    foo();
  }
  static int m;
#pragma omp target teams private(m) // OK
  foo();

  return 0;
}
