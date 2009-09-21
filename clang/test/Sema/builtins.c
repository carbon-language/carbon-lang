// RUN: clang-cc %s -fsyntax-only -verify -pedantic -triple=i686-apple-darwin9
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
void test7() {
  CFSTR("\242");
  CFSTR("\0"); // expected-warning {{ CFString literal contains NUL character }}
  CFSTR(242); // expected-error {{ CFString literal is not a string constant }} expected-warning {{incompatible integer to pointer conversion}}
  CFSTR("foo", "bar"); // expected-error {{too many arguments to function call}}
}


// atomics.

void test9(short v) {
  unsigned i, old;
  
  old = __sync_fetch_and_add();  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add(&old);  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add((int**)0, 42i); // expected-warning {{imaginary constants are an extension}}
}


// rdar://7236819
void test10(void) __attribute__((noreturn));

void test10(void) {
  __asm__("int3");  
  __builtin_unreachable();
 
  // No warning about falling off the end of a noreturn function.
}
