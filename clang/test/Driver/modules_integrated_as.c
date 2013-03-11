// RUN: %clang -fmodules -no-integrated-as -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error: modules can only be used with the compiler's integrated assembler
// CHECK note: '-no-integrated-as' cannot be used with '-fmodules'
