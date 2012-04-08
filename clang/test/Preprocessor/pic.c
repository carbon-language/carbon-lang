// RUN: %clang -target i386-unknown-unknown -static -dM -E -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC-NOT: #define __PIC__
// CHECK-STATIC-NOT: #define __pic__
// RUN: %clang -target i386-unknown-unknown -fpic -dM -E -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-LOWERPIC %s
// CHECK-LOWERPIC: #define __PIC__ 1
// CHECK-LOWERPIC: #define __pic__ 1
// RUN: %clang -target i386-unknown-unknown -fPIC -dM -E -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-UPPERPIC %s
// CHECK-UPPERPIC: #define __PIC__ 2
// CHECK-UPPERPIC: #define __pic__ 2
