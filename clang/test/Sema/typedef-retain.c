// RUN: clang -fsyntax-only -verify %s

typedef float float4 __attribute__((vector_size(16)));
typedef int int4 __attribute__((vector_size(16)));
typedef int4* int4p;

void test1(float4 a, int4 *result, int i) {
    result[i] = a; // expected-error {{assigning 'float4', expected 'int4'}}
}

void test2(float4 a, int4p result, int i) {
    result[i] = a; // expected-error {{assigning 'float4', expected 'int4'}}
}

// PR2039
typedef int a[5];
void z() {
  typedef const a b;
  b r;
  r[0]=10;  // expected-error {{read-only variable is not assignable}}
}

int e(const a y) {
  y[0] = 10; // expected-error {{read-only variable is not assignable}}
}

