// RUN: %clang_cc1 -E -dM -triple thumbv7m-apple-unknown-macho %s | FileCheck %s

// CHECK: #define __APPLE_CC__
// CHECK: #define __APPLE__
// CHECK-NOT: #define __MACH__
