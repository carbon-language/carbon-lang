// RUN: %clang_cc1 -fsyntax-only -Wnonnull -Wnullability %s -verify
// rdar://19160762

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif


int * __nullable foo(int * __nonnull x);

int *__nonnull ret_nonnull();

int *foo(int *x) {
  return 0;
}

int * __nullable foo1(int * __nonnull x); // expected-note {{previous declaration is here}}

int *foo1(int * __nullable x) { // expected-warning {{nullability specifier '__nullable' conflicts with existing specifier '__nonnull'}}
  return 0;
}

int * __nullable foo2(int * __nonnull x);

int *foo2(int * __nonnull x) {
  return 0;
}

int * __nullable foo3(int * __nullable x); // expected-note {{previous declaration is here}}

int *foo3(int * __nonnull x) { // expected-warning {{nullability specifier '__nonnull' conflicts with existing specifier '__nullable'}}
  return 0;
}

int * ret_nonnull() {
  return 0; // expected-warning {{null returned from function that requires a non-null return value}}
}

int main () {
  foo(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
