// Make sure AddressSanitizer disables _FORTIFY_SOURCE on Darwin.

// REQUIRES: system-darwin
// RUN: %clang -faddress-sanitizer %s -E -dM -o - | FileCheck %s
// CHECK: #define _FORTIFY_SOURCE 0
