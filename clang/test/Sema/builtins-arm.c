// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify -DTEST0 %s
// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify -DTEST1 %s

#ifdef TEST0
void __clear_cache(char*, char*);
#endif

#ifdef TEST1
void __clear_cache(void*, void*);
#endif

// va_list on ARM is void*.
void test2() {
  __builtin_va_list ptr = "x";
  *ptr = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}
