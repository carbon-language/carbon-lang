// RUN: %clang -### -c -save-temps -integrated-as %s 2>&1 | FileCheck %s

// gcc is invoked instead of clang-cc1as with gcc-driver -save-temps.
// REQUIRES: clang-driver

// CHECK: cc1as
// CHECK: -relax-all
