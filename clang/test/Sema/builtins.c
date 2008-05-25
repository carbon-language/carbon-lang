// RUN: clang %s -fsyntax-only -verify -pedantic -triple=i686-apple-darwin9
// This test needs to set the target because it uses __builtin_ia32_vec_ext_v4si

int test1(float a, int b) {
  return __builtin_isless(a, b);
}
int test2(int a, int b) {
  return __builtin_islessequal(a, b);  // expected-error {{floating point type}}
}

int test3(double a, float b) {
  return __builtin_isless(a, b);
}
int test4(int* a, double b) {
  return __builtin_islessequal(a, b);  // expected-error {{floating point type}}
}

int test5(float a, long double b) {
  return __builtin_isless(a, b, b);  // expected-error {{too many arguments}}
}
int test6(float a, long double b) {
  return __builtin_islessequal(a);  // expected-error {{too few arguments}}
}


#define CFSTR __builtin___CFStringMakeConstantString
void cfstring() {
  CFSTR("\242"); // expected-warning {{ CFString literal contains non-ASCII character }}
  CFSTR("\0"); // expected-warning {{ CFString literal contains NUL character }}
  CFSTR(242); // expected-error {{ CFString literal is not a string constant }} expected-warning {{incompatible integer to pointer conversion}}
  CFSTR("foo", "bar"); // expected-error {{ error: too many arguments to function }}
}


typedef __attribute__(( ext_vector_type(16) )) unsigned char uchar16;  // expected-warning {{extension}}

// rdar://5905347
unsigned char foo( short v ) {
  uchar16 c;
  return __builtin_ia32_vec_ext_v4si( c );  // expected-error {{too few arguments to function}}
}

