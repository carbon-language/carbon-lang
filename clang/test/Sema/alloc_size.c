// RUN: %clang_cc1 -fsyntax-only -verify %s

void* my_malloc(unsigned char) __attribute__((alloc_size(1)));
void* my_calloc(unsigned char, short) __attribute__((alloc_size(1,2)));
void* my_realloc(void*, unsigned) __attribute__((alloc_size(2)));


void* fn1(int) __attribute__((alloc_size("xpto"))); // expected-error{{attribute requires integer constant}}

void* fn2(void*) __attribute__((alloc_size(1))); // expected-error{{attribute requires integer constant}}

void* fn3(unsigned) __attribute__((alloc_size(0))); // expected-error{{attribute parameter 1 is out of bounds}}
void* fn4(unsigned) __attribute__((alloc_size(2))); // expected-error{{attribute parameter 1 is out of bounds}}

void fn5(unsigned) __attribute__((alloc_size(1))); // expected-warning{{only applies to functions that return a pointer}}
char fn6(unsigned) __attribute__((alloc_size(1))); // expected-warning{{only applies to functions that return a pointer}}

void* fn7(unsigned) __attribute__((alloc_size)); // expected-error {{attribute takes at least 1 argument}}

void *fn8(int, int) __attribute__((alloc_size(1, 1))); // expected-error {{attribute parameter 2 is duplicated}}

void* fn9(unsigned) __attribute__((alloc_size(12345678901234567890123))); // expected-warning {{integer constant is too large for its type}} // expected-error {{attribute parameter 1 is out of bounds}}
