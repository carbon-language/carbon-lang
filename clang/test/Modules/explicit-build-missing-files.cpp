// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'extern int a;' > %t/a.h
// RUN: echo 'extern int b; template<typename T> int b2 = T::error;' > %t/b.h
// RUN: echo 'module a { header "a.h" header "b.h" }' > %t/modulemap

// We lazily check that the files referenced by an explicitly-specified .pcm
// file exist. Test this by removing files and ensuring that the compilation
// still succeeds.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module -fmodule-name=a -x c++ %t/modulemap -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: rm %t/modulemap
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: rm %t/b.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s --check-prefix=MISSING-B
// RUN: rm %t/a.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -verify

#include "a.h" // expected-error {{file not found}}
int x = b;

#ifdef ERRORS
int y = b2<int>;
// CHECK: In module 'a':
// CHECK-NEXT: b.h:1:45: error:

// MISSING-B: could not find file '{{.*}}b.h'
// MISSING-B-NOT: please delete the module cache
#endif
