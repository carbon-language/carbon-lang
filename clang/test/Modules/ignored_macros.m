// First trial: pass -DIGNORED=1 to both. It should be ignored in both
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodule-cache-path %t.modules -DIGNORED=1 -fmodules -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: %clang_cc1 -fmodule-cache-path %t.modules -DIGNORED=1 -fmodules -I %S/Inputs -include-pch %t.pch %s -verify

// Second trial: pass -DIGNORED=1 only to the second invocation.
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodule-cache-path %t.modules -fmodules -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: not %clang_cc1 -fmodule-cache-path %t.modules -DIGNORED=1 -fmodules -I %S/Inputs -include-pch %t.pch %s > %t.err 2>&1
// RUN: FileCheck -check-prefix=CHECK-CONFLICT %s < %t.err
// CHECK-CONFLICT: module 'ignored_macros' found in both

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
@import ignored_macros;
#endif

@import ignored_macros;

struct Point p;
