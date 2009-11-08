// RUN: clang-cc %s -Eonly -Werror=undef -verify
// RUN: clang-cc %s -Eonly -Werror-undef -verify

extern int x;

#if foo   // expected-error {{'foo' is not defined, evaluates to 0}}
#endif

#ifdef foo
#endif

#if defined(foo)
#endif


// PR3938
#if 0
#ifdef D
#else 1       // Should not warn due to C99 6.10p4
#endif
#endif
