// RUN: %clang -S -v -o %t %s        2>&1 | FileCheck %s
// CHECK-NOT: -g

// RUN: %clang -S -v -o %t %s -g     2>&1 | FileCheck %s
// CHECK: -g

// RUN: %clang -S -v -o %t %s -g0    2>&1 | FileCheck %s
// CHECK-NOT: -g

// RUN: %clang -S -v -o %t %s -g -g0 2>&1 | FileCheck %s
// CHECK-NOT: -g

// RUN: %clang -S -v -o %t %s -g0 -g 2>&1 | FileCheck %s
// CHECK: -g
