// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify -fshort-wchar %s

#include <stdint.h>

const bool swchar = (wchar_t)-1 > (wchar_t)0;

#ifdef __WCHAR_UNSIGNED__
int signed_test[!swchar];
#else
int signed_test[swchar];
#endif

int max_test[WCHAR_MAX == (swchar ? -(WCHAR_MIN+1) : (wchar_t)-1)];
int min_test[WCHAR_MIN == (swchar ? 0 : -WCHAR_MAX-1)];
