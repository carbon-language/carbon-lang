// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'extern int a;' > %t/a.h
// RUN: echo 'extern int b;' > %t/b.h
// RUN: echo 'module a { header "a.h" header "b.h" }' > %t/modulemap

// We lazily check that the files referenced by an explicitly-specified .pcm
// file exist. Test this by removing files and ensuring that the compilation
// still succeeds.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module -fmodule-name=a -x c++ %t/modulemap -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: rm %t/modulemap
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: rm %t/b.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: rm %t/a.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -verify

#include "a.h" // expected-error {{file not found}}
int x = b;
