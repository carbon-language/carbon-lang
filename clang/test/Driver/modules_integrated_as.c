// RUN: %clang -fsyntax-only modules_integrated_as.c -fmodules -no-integrated-as -### 2>&1 | FileCheck %s

// Test that the autolinking feature is disabled with *not* using the
// integrated assembler.

// CHECK-NOT: -fmodules-autolink
