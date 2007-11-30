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
  v1 = v3; // expected-error {{incompatible types assigning 'v1s' to 'v2s'}}
  v1 = v4; // expected-error {{incompatible types assigning 'v2f' to 'v2s'}}
  v1 = v5;
  
  v2 = v1;
  v2 = v3; // expected-error {{incompatible types assigning 'v1s' to 'v2u'}}
  v2 = v4; // expected-error {{incompatible types assigning 'v2f' to 'v2u'}}
  v2 = v5;
  
  v3 = v1; // expected-error {{incompatible types assigning 'v2s' to 'v1s'}}
  v3 = v2; // expected-error {{incompatible types assigning 'v2u' to 'v1s'}}
  v3 = v4; // expected-error {{incompatible types assigning 'v2f' to 'v1s'}}
  v3 = v5; // expected-error {{incompatible types assigning 'v4ss' to 'v1s'}}
  
  v4 = v1; // expected-error {{incompatible types assigning 'v2s' to 'v2f'}}
  v4 = v2; // expected-error {{incompatible types assigning 'v2u' to 'v2f'}}
  v4 = v3; // expected-error {{incompatible types assigning 'v1s' to 'v2f'}}
  v4 = v5; // expected-error {{incompatible types assigning 'v4ss' to 'v2f'}}
  
  v5 = v1;
  v5 = v2;
  v5 = v3; // expected-error {{incompatible types assigning 'v1s' to 'v4ss'}}
  v5 = v4; // expected-error {{incompatible types assigning 'v2f' to 'v4ss'}}
}
