// We should be synthesizing properties by default on all platforms now.
// RUN: %clang -### -target armv7-unknown-freebsd %s 2>&1 | FileCheck %s
// RUN: %clang -### -target armv7-apple-ios %s 2>&1 | FileCheck %s
// RUN: %clang -### -target i686-apple-macosx %s 2>&1 | FileCheck %s
// REQUIRES: clang-driver
// CHECK: -fobjc-default-synthesize
