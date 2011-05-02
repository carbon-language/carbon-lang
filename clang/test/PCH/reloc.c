// RUN: %clang_cc1 -emit-pch -o %t -relocatable-pch -isysroot %S/libroot %S/libroot/usr/include/reloc.h
// RUN: %clang_cc1 -include-pch %t -isysroot %S/libroot %s -verify
// RUN: not %clang_cc1 -include-pch %t %s

#include <reloc.h>

int x = 2; // expected-error{{redefinition}}
int y = 5; // expected-error{{redefinition}}




// expected-note{{previous definition}}
// expected-note{{previous definition}}
