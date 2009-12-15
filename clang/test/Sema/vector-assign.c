// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversions
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef signed int v2s __attribute__ ((vector_size (8)));
typedef signed int v1s __attribute__ ((vector_size (4)));
typedef float v2f __attribute__ ((vector_size(8)));
typedef signed short v4ss __attribute__ ((vector_size (8)));

void test1() {
  v2s v1;
  v2u v2;
  v1s v3;
  v2f v4;
  v4ss v5;
  
  v1 = v2; // expected-warning {{incompatible vector types assigning 'v2u', expected 'v2s'}}
  v1 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2s'}}
  v1 = v4; // expected-warning {{incompatible vector types assigning 'v2f', expected 'v2s'}}
  v1 = v5; // expected-warning {{incompatible vector types assigning 'v4ss', expected 'v2s'}}
  
  v2 = v1; // expected-warning {{incompatible vector types assigning 'v2s', expected 'v2u'}}
  v2 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2u'}}
  v2 = v4; // expected-warning {{incompatible vector types assigning 'v2f', expected 'v2u'}}
  v2 = v5; // expected-warning {{incompatible vector types assigning 'v4ss', expected 'v2u'}}
  
  v3 = v1; // expected-error {{incompatible type assigning 'v2s', expected 'v1s'}}
  v3 = v2; // expected-error {{incompatible type assigning 'v2u', expected 'v1s'}}
  v3 = v4; // expected-error {{incompatible type assigning 'v2f', expected 'v1s'}}
  v3 = v5; // expected-error {{incompatible type assigning 'v4ss', expected 'v1s'}}
  
  v4 = v1; // expected-warning {{incompatible vector types assigning 'v2s', expected 'v2f'}}
  v4 = v2; // expected-warning {{incompatible vector types assigning 'v2u', expected 'v2f'}}
  v4 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2f'}}
  v4 = v5; // expected-warning {{incompatible vector types assigning 'v4ss', expected 'v2f'}}
  
  v5 = v1; // expected-warning {{incompatible vector types assigning 'v2s', expected 'v4ss'}}
  v5 = v2; // expected-warning {{incompatible vector types assigning 'v2u', expected 'v4ss'}}
  v5 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v4ss'}}
  v5 = v4; // expected-warning {{incompatible vector types assigning 'v2f', expected 'v4ss'}}
}

// PR2263
float test2(__attribute__((vector_size(16))) float a, int b) {
   return a[b];
}

// PR4838
typedef long long __attribute__((__vector_size__(2 * sizeof(long long))))
longlongvec;

void test3a(longlongvec *);
void test3(const unsigned *src) {
  test3a(src);  // expected-warning {{incompatible pointer types passing 'unsigned int const *', expected 'longlongvec *'}}
}
