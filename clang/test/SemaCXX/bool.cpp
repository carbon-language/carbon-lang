// RUN: clang-cc -fsyntax-only -verify %s 

// Bool literals can be enum values.
enum {
  ReadWrite = false,
  ReadOnly = true
};

// bool cannot be decremented, and gives a warning on increment
void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
  b++; // expected-warning {{incrementing expression of type bool is deprecated}}
  --b; // expected-error {{cannot decrement expression of type bool}}
  b--; // expected-error {{cannot decrement expression of type bool}}

  bool *b1 = (int *)0; // expected-error{{expected 'bool *'}}
}

// static_assert_arg_is_bool(x) compiles only if x is a bool.
template <typename T>
void static_assert_arg_is_bool(T x) {
  bool* p = &x;
}

void test2() {
  int n = 2;
  static_assert_arg_is_bool(n && 4);
  static_assert_arg_is_bool(n || 5);
}
