// RUN: clang-cc -emit-pch -o %t --relocatable-pch -isysroot `pwd`/libroot `pwd`/libroot/usr/include/reloc.h &&
// RUN: clang-cc -include-pch %t -isysroot `pwd`/libroot %s -verify
// FIXME (test harness can't do this?): not clang-cc -include-pch %t %s

#include <reloc.h>

int x = 2; // expected-error{{redefinition}}
int y = 5; // expected-error{{redefinition}}




// expected-note{{previous definition}}
// expected-note{{previous definition}}