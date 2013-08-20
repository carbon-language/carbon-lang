// RUN: %clang -### -c -save-temps -integrated-as %s 2>&1 | FileCheck %s

// CHECK: cc1as
// CHECK: -mrelax-all
