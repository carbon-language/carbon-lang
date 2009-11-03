// RUN: clang-cc -dM -undef -Dfoo=1 -E %s | FileCheck %s

// CHECK-NOT: #define __clang__
// CHECK: #define foo 1
