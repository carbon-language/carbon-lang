// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int I>
struct TS {
  __attribute__((returns_nonnull))
  void *value_dependent(void) {
    return I; // no-warning
  }

  __attribute__((returns_nonnull))
  void *value_independent(void) {
    return 0; // expected-warning {{null returned from function that requires a non-null return value}}
  }
};

