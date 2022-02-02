// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -std=c++11 %s -Wuninitialized

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

void xxx(int argc) {
  int fp; // expected-note {{initialize the variable 'fp' to silence this warning}}
#pragma omp target teams distribute reduction(+:fp) // expected-warning {{variable 'fp' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void foobar(int &ref) {
#pragma omp target teams distribute reduction(+:ref)
  for (int j=0; j<100; j++) foo();
}

struct S1; // expected-note {{declared here}} expected-note 4 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
  S2 &operator+(const S2 &arg) { return (*this); } // expected-note 3 {{implicitly declared private here}}

public:
  S2() : a(0) {}
  S2(S2 &s2) : a(s2.a) {}
  static float S2s; // expected-note 2 {{static data member is predetermined as shared}}
  static const float S2sc; // expected-note 2 {{'S2sc' declared here}}
};
const float S2::S2sc = 0;
S2 b;                     // expected-note 3 {{'b' defined here}}
const S2 ba[5];           // expected-note 2 {{'ba' defined here}}
class S3 {
  int a;

public:
  int b;
  S3() : a(0) {}
  S3(const S3 &s3) : a(s3.a) {}
  S3 operator+(const S3 &arg1) { return arg1; }
};
int operator+(const S3 &arg1, const S3 &arg2) { return 5; }
S3 c;               // expected-note 3 {{'c' defined here}}
const S3 ca[5];     // expected-note 2 {{'ca' defined here}}
extern const int f; // expected-note 4 {{'f' declared here}}
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}
  S4(const S4 &s4);
  S4 &operator+(const S4 &arg) { return (*this); }

public:
  S4(int v) : a(v) {}
};
S4 &operator&=(S4 &arg1, S4 &arg2) { return arg1; }
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}}
  S5(const S5 &s5) : a(s5.a) {}
  S5 &operator+(const S5 &arg);

public:
  S5(int v) : a(v) {}
};
class S6 { // expected-note 3 {{candidate function (the implicit copy assignment operator) not viable: no known conversion from 'int' to 'const S6' for 1st argument}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 3 {{candidate function (the implicit move assignment operator) not viable}}
#endif
  int a;

public:
  S6() : a(6) {}
  operator int() { return 6; }
} o;

S3 h, k;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class T>       // expected-note {{declared here}}
T tmain(T argc) {
  const T d = T();       // expected-note 4 {{'d' defined here}}
  const T da[5] = {T()}; // expected-note 2 {{'da' defined here}}
  T qa[5] = {T()};
  T i, z;
  T &j = i;                    // expected-note 4 {{'j' defined here}}
  S3 &p = k;                   // expected-note 2 {{'p' defined here}}
  const T &r = da[(int)i];     // expected-note 2 {{'r' defined here}}
  T &q = qa[(int)i];           // expected-note 2 {{'q' defined here}}
  T fl;
#pragma omp target teams distribute reduction // expected-error {{expected '(' after 'reduction'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction + // expected-error {{expected '(' after 'reduction'}} expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction( // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(- // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction() // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(*) // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(\) // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(& : argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid operands to binary expression ('float' and 'float')}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(| : argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid operands to binary expression ('float' and 'float')}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(|| : argc ? i : argc) // expected-error 2 {{expected variable name, array element or array section}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(foo : argc) //expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'float'}} expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(^ : T) // expected-error {{'T' does not refer to a value}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : z, a, b, c, d, f) // expected-error {{a reduction list item with incomplete type 'S1'}} expected-error 3 {{const-qualified variable cannot be reduction}} expected-error 2 {{'operator+' is a private member of 'S2'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(min : a, b, c, d, f) // expected-error {{a reduction list item with incomplete type 'S1'}} expected-error 4 {{arguments of OpenMP clause 'reduction' for 'min' or 'max' must be of arithmetic type}} expected-error 3 {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(max : h.b) // expected-error {{expected variable name, array element or array section}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : ba) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(* : ca) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(- : da) // expected-error {{const-qualified variable cannot be reduction}} expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(^ : fl) // expected-error {{invalid operands to binary expression ('float' and 'float')}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : S2::S2s) // expected-error {{shared variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : S2::S2sc) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : h, k) // expected-error {{threadprivate or thread local variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : o) // expected-error 2 {{no viable overloaded '='}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute private(i), reduction(+ : j), reduction(+ : q) // expected-error 4 {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp parallel private(k)
#pragma omp target teams distribute reduction(+ : p), reduction(+ : p) // expected-error 2 {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : p), reduction(+ : p) // expected-error 2 {{variable can appear only once in OpenMP 'reduction' clause}} expected-note 2 {{previously referenced here}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : r) // expected-error 2 {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp parallel shared(i)
#pragma omp parallel reduction(min : i)
#pragma omp target teams distribute reduction(max : j) // expected-error 2 {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute uses_allocators(omp_thread_mem_alloc) allocate(omp_thread_mem_alloc: fl) reduction(+ : fl) // expected-warning 2 {{allocator with the 'thread' trait access has unspecified behavior on 'target teams distribute' directive}} omp45-error {{unexpected OpenMP clause 'uses_allocators' in directive '#pragma omp target teams distribute'}}
    for (int j=0; j<100; j++) foo();

  return T();
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5;       // expected-note 2 {{'d' defined here}}
  const int da[5] = {0}; // expected-note {{'da' defined here}}
  int qa[5] = {0};
  S4 e(4);
  S5 g(5);
  int i, z;
  int &j = i;                  // expected-note 2 {{'j' defined here}}
  S3 &p = k;                   // expected-note 2 {{'p' defined here}}
  const int &r = da[i];        // expected-note {{'r' defined here}}
  int &q = qa[i];              // expected-note {{'q' defined here}}
  float fl;
#pragma omp target teams distribute reduction // expected-error {{expected '(' after 'reduction'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction + // expected-error {{expected '(' after 'reduction'}} expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction( // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(- // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction() // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(*) // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(\) // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(foo : argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{incorrect reduction identifier, expected one of '+', '-', '*', '&', '|', '^', '&&', '||', 'min' or 'max'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(| : argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(|| : argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name, array element or array section}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(~ : argc) // expected-error {{expected unqualified-id}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : argc)
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(^ : S1) // expected-error {{'S1' does not refer to a value}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : z, a, b, c, d, f) // expected-error {{a reduction list item with incomplete type 'S1'}} expected-error 2 {{const-qualified variable cannot be reduction}} expected-error {{'operator+' is a private member of 'S2'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(min : a, b, c, d, f) // expected-error {{a reduction list item with incomplete type 'S1'}} expected-error 2 {{arguments of OpenMP clause 'reduction' for 'min' or 'max' must be of arithmetic type}} expected-error 2 {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(max : h.b) // expected-error {{expected variable name, array element or array section}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : ba) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(* : ca) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(- : da) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(^ : fl) // expected-error {{invalid operands to binary expression ('float' and 'float')}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : S2::S2s) // expected-error {{shared variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(&& : S2::S2sc) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(& : e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}} expected-error {{invalid operands to binary expression ('S5' and 'S5')}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : h, k, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : o) // expected-error {{no viable overloaded '='}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute private(i), reduction(+ : j), reduction(+ : q) // expected-error 2 {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp parallel private(k)
#pragma omp target teams distribute reduction(+ : p), reduction(+ : p) // expected-error 2 {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : p), reduction(+ : p) // expected-error {{variable can appear only once in OpenMP 'reduction' clause}} expected-note {{previously referenced here}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : r) // expected-error {{const-qualified variable cannot be reduction}}
  for (int j=0; j<100; j++) foo();
#pragma omp parallel shared(i)
#pragma omp parallel reduction(min : i)
#pragma omp target teams distribute reduction(max : j) // expected-error {{argument of OpenMP clause 'reduction' must reference the same object in all threads}}
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(+ : fl)
    for (int j=0; j<100; j++) foo();
  static int m;
#pragma omp target teams distribute reduction(+ : m) // OK
  for (int j=0; j<100; j++) foo();
#pragma omp target teams distribute reduction(task, + : m) // omp45-error 2 {{expected expression}} omp45-warning {{missing ':' after reduction identifier - ignoring}} omp50-error {{'reduction' clause with 'task' modifier allowed only on non-simd parallel or worksharing constructs}}
  for (int j=0; j<100; j++) foo();

  return tmain(argc) + tmain(fl); // expected-note {{in instantiation of function template specialization 'tmain<int>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<float>' requested here}}
}
