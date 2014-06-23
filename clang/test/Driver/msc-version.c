// RUN: %clang -target i686-windows -fms-compatibility -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-NO-MSC-VERSION

// CHECK-NO-MSC-VERSION: _MSC_BUILD 1
// CHECK-NO-MSC-VERSION: _MSC_FULL_VER 170000000
// CHECK-NO-MSC-VERSION: _MSC_VER 1700

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=1600 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION

// CHECK-MSC-VERSION: _MSC_BUILD 1
// CHECK-MSC-VERSION: _MSC_FULL_VER 160000000
// CHECK-MSC-VERSION: _MSC_VER 1600

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=160030319 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-EXT

// CHECK-MSC-VERSION-EXT: _MSC_BUILD 1
// CHECK-MSC-VERSION-EXT: _MSC_FULL_VER 160030319
// CHECK-MSC-VERSION-EXT: _MSC_VER 1600

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=14 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR

// CHECK-MSC-VERSION-MAJOR: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR: _MSC_FULL_VER 140000000
// CHECK-MSC-VERSION-MAJOR: _MSC_VER 1400

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=17.00 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR

// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_FULL_VER 170000000
// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_VER 1700

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=15.00.20706 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR-BUILD

// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_FULL_VER 150020706
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_VER 1500

// RUN: %clang -target i686-windows -fms-compatibility -fmsc-version=15.00.20706.01 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH

// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_FULL_VER 150020706
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_VER 1500

