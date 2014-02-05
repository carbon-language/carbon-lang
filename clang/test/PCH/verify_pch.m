// Precompile
// RUN: cp %s %t.h
// RUN: %clang_cc1 -x objective-c-header -emit-pch -o %t.pch %t.h

// Verify successfully
// RUN: %clang_cc1 -x objective-c -verify-pch %t.pch

// Incompatible lang options ignored
// RUN: %clang_cc1 -x objective-c -fno-builtin -verify-pch %t.pch

// Stale dependency
// RUN: echo ' ' >> %t.h
// RUN: not %clang_cc1 -x objective-c -verify-pch %t.pch 2> %t.log.2
// RUN: FileCheck -check-prefix=CHECK-STALE-DEP %s < %t.log.2
// CHECK-STALE-DEP: file '{{.*}}.h' has been modified since the precompiled header '{{.*}}.pch' was built
