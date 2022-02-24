// RUN: %clang_cc1 -x c++ -std=c++11 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -x c++ -std=c++11 -verify -fopenmp-simd %s -Wuninitialized

struct B {
  static int ib[20]; // expected-note 0 {{'B::ib' declared here}}
  static constexpr int bfoo() { return 8; }
};
namespace X {
  B x; // expected-note {{'x' defined here}}
};
constexpr int bfoo() { return 4; }

int **z;
const int C1 = 1;
const int C2 = 2;
void test_aligned_colons(int *&rp)
{
  int *B = 0;
  #pragma omp master taskloop simd aligned(B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'}}
  #pragma omp master taskloop simd aligned(B::ib:B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd aligned(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'?}}
  #pragma omp master taskloop simd aligned(z:B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd aligned(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'int **'}}
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'B'}}
  #pragma omp master taskloop simd aligned(X::x : ::z)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'B'}}
  #pragma omp master taskloop simd aligned(B,rp,::z: X::x)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd aligned(::z)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp master taskloop simd aligned(B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-warning@+1 {{aligned clause will be ignored because the requested alignment is not a power of 2}}
  #pragma omp master taskloop simd aligned(B::ib,B:C1+C2)
  for (int i = 0; i < 10; ++i) ;
}

// expected-note@+1 {{'num' defined here}}
template<int L, class T, class N> T test_template(T* arr, N num) {
  N i;
  T sum = (T)0;
  T ind2 = - num * L;
  // Negative number is passed as L.
  // expected-error@+1 {{argument to 'aligned' clause must be a strictly positive integer value}}
  #pragma omp master taskloop simd aligned(arr:L)
  for (i = 0; i < num; ++i) {
    T cur = arr[(int)ind2];
    ind2 += L;
    sum += cur;
  }
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'int'}}
  #pragma omp master taskloop simd aligned(num:4)
  for (i = 0; i < num; ++i);
  return T();
}

template<int LEN> int test_warn() {
  int *ind2 = 0;
  // expected-error@+1 {{argument to 'aligned' clause must be a strictly positive integer value}}
  #pragma omp master taskloop simd aligned(ind2:LEN)
  for (int i = 0; i < 100; i++) {
    ind2 += LEN;
  }
  return 0;
}

struct S1; // expected-note 2 {{declared here}}
extern S1 a; // expected-note {{'a' declared here}}
class S2 {
  mutable int a;
public:
  S2():a(0) { }
};
const S2 b; // expected-note 1 {{'b' defined here}}
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 ca[5];
class S4 {
  int a;
  S4();
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
public:
  S5(int v):a(v) { }
};

S3 h; // expected-note 2 {{'h' defined here}}
#pragma omp threadprivate(h)

template<class I, class C> int foomain(I argc, C **argv) {
  I e(argc);
  I g(argc);
  int i; // expected-note {{'i' defined here}}
  // expected-note@+1 {{declared here}}
  int &j = i;
  #pragma omp master taskloop simd aligned // expected-error {{expected '(' after 'aligned'}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned () // expected-error {{expected expression}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (I k = 0; k < argc; ++k) ++k;
// FIXME: Should argc really be a pointer?
  #pragma omp master taskloop simd aligned (*argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argc : 5) // expected-warning {{aligned clause will be ignored because the requested alignment is not a power of 2}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (S1) // expected-error {{'S1' does not refer to a value}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argv[1]) // expected-error {{expected variable name}}
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned(e, g)
  for (I k = 0; k < argc; ++k) ++k;
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'S3'}}
  #pragma omp master taskloop simd aligned(h)
  for (I k = 0; k < argc; ++k) ++k;
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'int'}}
  #pragma omp master taskloop simd aligned(i)
  for (I k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int *v = 0;
    I i;
    #pragma omp master taskloop simd aligned(v:16)
    for (I k = 0; k < argc; ++k) { i = k; v += 2; }
  }
  float *f;
  #pragma omp master taskloop simd aligned(f)
  for (I k = 0; k < argc; ++k) ++k;
  int v = 0;
  // expected-note@+2 {{initializer of 'j' is not a constant expression}}
  // expected-error@+1 {{integral constant expression}}
  #pragma omp master taskloop simd aligned(f:j)
  for (I k = 0; k < argc; ++k) { ++k; v += j; }
  #pragma omp master taskloop simd aligned(f)
  for (I k = 0; k < argc; ++k) ++k;
  return 0;
}

// expected-note@+1 2 {{'argc' defined here}}
int main(int argc, char **argv) {
  double darr[100];
  // expected-note@+1 {{in instantiation of function template specialization 'test_template<-4, double, int>' requested here}}
  test_template<-4>(darr, 4);
  test_warn<4>(); // ok
  // expected-note@+1 {{in instantiation of function template specialization 'test_warn<0>' requested here}}
  test_warn<0>();

  int i;
  int &j = i;
  #pragma omp master taskloop simd aligned // expected-error {{expected '(' after 'aligned'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argv // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'int'}}
  #pragma omp master taskloop simd aligned (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'int'}}
  #pragma omp master taskloop simd aligned (argc)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'S1'}}
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'S2'}}
#pragma omp master taskloop simd aligned(a, b)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd aligned (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+1 {{argument of aligned clause should be array, pointer, reference to array or reference to pointer, not 'S3'}}
  #pragma omp master taskloop simd aligned(h)
  for (int k = 0; k < argc; ++k) ++k;
  int *pargc = &argc;
  // expected-note@+1 {{in instantiation of function template specialization 'foomain<int *, char>' requested here}}
  foomain<int*,char>(pargc,argv);
  return 0;
}

