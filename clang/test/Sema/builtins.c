// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -triple=i686-apple-darwin9
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
  const void *X;
  X = CFSTR("\242"); // expected-warning {{input conversion stopped}}
  X = CFSTR("\0"); // expected-warning {{ CFString literal contains NUL character }}
  X = CFSTR(242); // expected-error {{ CFString literal is not a string constant }} expected-warning {{incompatible integer to pointer conversion}}
  X = CFSTR("foo", "bar"); // expected-error {{too many arguments to function call}}
}


// atomics.

void test9(short v) {
  unsigned i, old;

  old = __sync_fetch_and_add();  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add(&old);  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add((unsigned*)0, 42i); // expected-warning {{imaginary constants are an extension}}

  // PR7600: Pointers are implicitly casted to integers and back.
  void *old_ptr = __sync_val_compare_and_swap((void**)0, 0, 0);

  // Ensure the return type is correct even when implicit casts are stripped
  // away. This triggers an assertion while checking the comparison otherwise.
  if (__sync_fetch_and_add(&old, 1) == 1) {
  }
}


// rdar://7236819
void test10(void) __attribute__((noreturn));

void test10(void) {
  __asm__("int3");
  __builtin_unreachable();

  // No warning about falling off the end of a noreturn function.
}

void test11(int X) {
  switch (X) {
  case __builtin_eh_return_data_regno(0):  // constant foldable.
    break;
  }

  __builtin_eh_return_data_regno(X);  // expected-error {{argument to '__builtin_eh_return_data_regno' must be a constant integer}}
}

// PR5062
void test12(void) __attribute__((__noreturn__));
void test12(void) {
  __builtin_trap();  // no warning because trap is noreturn.
}

void test_unknown_builtin(int a, int b) {
  __builtin_foo(a, b); // expected-error{{use of unknown builtin}}
}

int test13() {
  __builtin_eh_return(0, 0); // no warning, eh_return never returns.
}

// <rdar://problem/8228293>
void test14() {
  int old;
  old = __sync_fetch_and_min((volatile int *)&old, 1);
}

// <rdar://problem/8336581>
void test15(const char *s) {
  __builtin_printf("string is %s\n", s);
}

// PR7885
int test16() {
  return __builtin_constant_p() + // expected-error{{too few arguments}}
         __builtin_constant_p(1, 2); // expected-error {{too many arguments}}
}

