// RUN: clang-cc %s -Eonly -Werror=undef -verify

extern int x;

#if foo   // expected-error {{'foo' is not defined, evaluates to 0}}
#endif

#ifdef foo
#endif

#if defined(foo)
#endif

