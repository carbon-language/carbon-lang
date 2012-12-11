// RUN: %clang -### -c -integrated-as %s 2>&1 | FileCheck %s

// REQUIRES: clang-driver

// CHECK: cc1as
// CHECK-NOT: -relax-all
