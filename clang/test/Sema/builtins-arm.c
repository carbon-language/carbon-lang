// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armv7 -target-abi apcs-gnu \
// RUN:   -fsyntax-only -verify %s

void f(void *a, void *b) {
  __clear_cache(); // expected-error {{too few arguments to function call, expected 2, have 0}} // expected-note {{'__clear_cache' is a builtin with type 'void (void *, void *)}}
  __clear_cache(a); // expected-error {{too few arguments to function call, expected 2, have 1}}
  __clear_cache(a, b);
}

void __clear_cache(char*, char*); // expected-error {{conflicting types for '__clear_cache'}}
void __clear_cache(void*, void*);

#if defined(__ARM_PCS) || defined(__ARM_EABI__)
// va_list on ARM AAPCS is struct { void* __ap }.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";
  *(ptr.__ap) = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}
#else
// va_list on ARM apcs-gnu is void*.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";  // expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
  *(ptr.__ap) = '0';// expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
}

void test2() {
  __builtin_va_list ptr = "x";
  *ptr = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}

void test3() {
  __builtin_arm_dsb(16); // expected-error {{argument should be a value from 0 to 15}}
  __builtin_arm_dmb(17); // expected-error {{argument should be a value from 0 to 15}}
  __builtin_arm_isb(18); // expected-error {{argument should be a value from 0 to 15}}
}

#endif
