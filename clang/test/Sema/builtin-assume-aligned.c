// RUN: %clang_cc1 -fsyntax-only -verify %s

int test1(int *a) {
  a = __builtin_assume_aligned(a, 32, 0ull);
  return a[0];
}

int test2(int *a) {
  a = __builtin_assume_aligned(a, 32, 0);
  return a[0];
}

int test3(int *a) {
  a = __builtin_assume_aligned(a, 32);
  return a[0];
}

int test4(int *a) {
  a = __builtin_assume_aligned(a, -32); // expected-error {{requested alignment is not a power of 2}}
  // FIXME: The line below produces {{requested alignment is not a power of 2}}
  // on i386-freebsd, but not on x86_64-linux (for example).
  // a = __builtin_assume_aligned(a, 1ULL << 63);
  return a[0];
}

int test5(int *a, unsigned *b) {
  a = __builtin_assume_aligned(a, 32, b); // expected-warning {{incompatible pointer to integer conversion passing 'unsigned int *' to parameter of type}}
  return a[0];
}

int test6(int *a) {
  a = __builtin_assume_aligned(a, 32, 0, 0); // expected-error {{too many arguments to function call, expected at most 3, have 4}}
  return a[0];
}

int test7(int *a) {
  a = __builtin_assume_aligned(a, 31); // expected-error {{requested alignment is not a power of 2}}
  return a[0];
}

int test8(int *a, int j) {
  a = __builtin_assume_aligned(a, j); // expected-error {{must be a constant integer}}
  return a[0];
}

