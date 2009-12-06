// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: echo 'void f0();' > %t.dir/t.h
// RUN: chmod 000 %t.dir/t.h
// RUN: clang-cc -verify -I %t.dir %s

// FIXME: Is there a way to test this on Windows?
// XFAIL: win32

#include "t.h" // expected-error {{Permission denied}}
int f0(void);
