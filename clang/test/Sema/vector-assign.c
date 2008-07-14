// RUN: clang %s -verify -fsyntax-only -flax-vector-conversions
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef signed int v2s __attribute__ ((vector_size (8)));
typedef signed int v1s __attribute__ ((vector_size (4)));
typedef float v2f __attribute__ ((vector_size(8)));
typedef signed short v4ss __attribute__ ((vector_size (8)));

void f() {
  v2s v1;
  v2u v2;
  v1s v3;
  v2f v4;
  v4ss v5;
  
  v1 = v2; 
  v1 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2s'}}
  v1 = v4; 
  v1 = v5;
  
  v2 = v1;
  v2 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2u'}}
  v2 = v4; 
  v2 = v5;
  
  v3 = v1; // expected-error {{incompatible type assigning 'v2s', expected 'v1s'}}
  v3 = v2; // expected-error {{incompatible type assigning 'v2u', expected 'v1s'}}
  v3 = v4; // expected-error {{incompatible type assigning 'v2f', expected 'v1s'}}
  v3 = v5; // expected-error {{incompatible type assigning 'v4ss', expected 'v1s'}}
  
  v4 = v1; 
  v4 = v2; 
  v4 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v2f'}}
  v4 = v5;
  
  v5 = v1;
  v5 = v2;
  v5 = v3; // expected-error {{incompatible type assigning 'v1s', expected 'v4ss'}}
  v5 = v4;
}

// PR2263
float f2(__attribute__((vector_size(16))) float a, int b) {
   return a[b];
}

