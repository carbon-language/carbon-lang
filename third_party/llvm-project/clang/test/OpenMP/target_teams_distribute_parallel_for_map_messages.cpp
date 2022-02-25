// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp -fno-openmp-extensions -fopenmp-version=45 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,lt51 -fopenmp -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,ge51 -fopenmp -fno-openmp-extensions -fopenmp-version=51 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized

// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp-simd -fno-openmp-extensions -fopenmp-version=45 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void xxx(int argc) {
  int map; // expected-note {{initialize the variable 'map' to silence this warning}}
#pragma omp target teams distribute parallel for map(tofrom: map) // expected-warning {{variable 'map' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note 2 {{declared here}} // expected-note 3 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

typedef int from;

template <typename T, int I> // expected-note {{declared here}}
T tmain(T argc) {
  const T d = 5;
  const T da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  T i, t[20];
  T &j = i;
  T *k = &j;
  T x;
  T y, z;
  T to, tofrom, always;
  const T (&l)[5] = da;


#pragma omp target teams distribute parallel for map // expected-error {{expected '(' after 'map'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map() // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to:) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(from: argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(x: y) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[-1:]) // expected-error 2 {{array section must be a subset of the original array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[:-1]) // expected-error 2 {{section length is evaluated to a negative value -1}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[true:true])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(tofrom: t[:I])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(T: a) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}} expected-error {{incomplete type 'S1' where a complete type is required}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(T) // expected-error {{'T' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
// ge50-error@+2 2 {{expected addressable lvalue in 'map' clause}}
// lt50-error@+1 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target teams distribute parallel for map(I)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2s)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2sc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(x, z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to: to)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to, x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to x) // expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
// ge50-error@+3 2 {{expected addressable lvalue in 'map' clause}}
// lt50-error@+2 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target teams distribute parallel for map(tofrom \
                                                     : argc > 0 ? x : y)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(ba)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(ca)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2s)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2sc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(e, g)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(k), map(k) // lt50-error 2 {{variable already marked as mapped in current construct}} lt50-note 2 {{used here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(k), map(k[:5]) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}} lt50-note 2 {{used here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da[:4])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target data map(k, j, l)                        // lt50-note 2 {{used here}}
#pragma omp target teams distribute parallel for map(k[:4]) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(j)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l) map(l[:5]) // lt50-error 2 {{variable already marked as mapped in current construct}} lt50-note 2 {{used here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target data map(k[:4], j, l[:5]) // lt50-note 2 {{used here}}
  {
#pragma omp target teams distribute parallel for map(k) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}}
    for (i = 0; i < argc; ++i)
      foo();
#pragma omp target teams distribute parallel for map(j)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l)
  for (i = 0; i < argc; ++i) foo();
}

#pragma omp target teams distribute parallel for map(always, tofrom: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(always: x) // expected-error {{missing map type}}
  for (i = 0; i < argc; ++i) foo();
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target teams distribute parallel for map(tofrom, always: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(always, tofrom: always, tofrom, x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  int *k = &j;
  int x;
  int y, z;
  int to, tofrom, always;
  const int (&l)[5] = da;

#pragma omp target teams distribute parallel for map // expected-error {{expected '(' after 'map'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map() // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to:) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(from: argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(x: y) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[-1:]) // expected-error {{array section must be a subset of the original array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l[true:true])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to: to)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to, x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(to x) // expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
// ge50-error@+3 {{expected addressable lvalue in 'map' clause}}
// lt50-error@+2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target teams distribute parallel for map(tofrom \
                                                     : argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(argv[1])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(ba, z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(ca)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2s)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(S2::S2sc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(e, g)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute parallel for map(k), map(k) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute parallel for map(k), map(k[:5]) // lt50-error {{pointer cannot be mapped along with a section derived from itself}} lt50-note {{used here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(da[:4])
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target data map(k, j, l)                        // lt50-note {{used here}}
#pragma omp target teams distribute parallel for map(k[:4]) // lt50-error {{pointer cannot be mapped along with a section derived from itself}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(j)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute parallel for map(l) map(l[:5]) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target data map(k[:4], j, l[:5]) // lt50-note {{used here}}
  {
#pragma omp target teams distribute parallel for map(k) // lt50-error {{pointer cannot be mapped along with a section derived from itself}}
    for (i = 0; i < argc; ++i)
      foo();
#pragma omp target teams distribute parallel for map(j)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(l)
  for (i = 0; i < argc; ++i) foo();
  }

#pragma omp target teams distribute parallel for map(always, tofrom: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(always: x) // expected-error {{missing map type}}
  for (i = 0; i < argc; ++i) foo();
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target teams distribute parallel for map(tofrom, always: x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(always, tofrom: always, tofrom, x)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for map(delete: j) // expected-error {{map type 'delete' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute parallel for map(release: j) // expected-error {{map type 'release' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i)
    foo();

  return tmain<int, 3>(argc)+tmain<from, 4>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
}

