// RUN: %clang_cc1 -dM -E -o - %s \
// RUN:   | FileCheck %s
// CHECK-NOT: #define __PIC__
// CHECK-NOT: #define __pic__
// RUN: %clang_cc1 -pic-level 1 -dM -E -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-PIC1 %s
// CHECK-PIC1: #define __PIC__ 1
// CHECK-PIC1: #define __pic__ 1
// RUN: %clang_cc1 -pic-level 2 -dM -E -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-PIC2 %s
// CHECK-PIC2: #define __PIC__ 2
// CHECK-PIC2: #define __pic__ 2
