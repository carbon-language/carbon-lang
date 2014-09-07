// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int z>
int test9(int *a) {
  a = (int *) __builtin_assume_aligned(a, z + 1); // expected-error {{requested alignment is not a power of 2}}
  return a[0];
}

void test9i(int *a) {
  test9<42>(a); // expected-note {{in instantiation of function template specialization 'test9<42>' requested here}}
}

template<typename T>
int test10(int *a, T z) {
  a = (int *) __builtin_assume_aligned(a, z + 1); // expected-error {{must be a constant integer}}
  return a[0];
}

int test10i(int *a) {
  return test10(a, 42); // expected-note {{in instantiation of function template specialization 'test10<int>' requested here}}
}

