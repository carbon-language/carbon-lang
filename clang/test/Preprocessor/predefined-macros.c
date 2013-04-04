// This test verifies that the correct macros are predefined.
//
// RUN: %clang_cc1 %s -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fmsc-version=1300 -o - | FileCheck %s --check-prefix=CHECK-MS
// CHECK-MS: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS: #define _MSC_EXTENSIONS 1
// CHECK-MS: #define _MSC_VER 1300
// CHECK-MS: #define _M_IX86 600
// CHECK-MS: #define _M_IX86_FP
// CHECK-MS: #define _WIN32 1
// CHECK-MS-NOT: #define __GNUC__
//
// RUN: %clang_cc1 %s -E -dM -ffast-math -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FAST-MATH
// CHECK-FAST-MATH: #define __FAST_MATH__
// CHECK-FAST-MATH: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang_cc1 %s -E -dM -ffinite-math-only -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FINITE-MATH-ONLY
// CHECK-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang %s -E -dM -fno-finite-math-only -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-FINITE-MATH-ONLY
// CHECK-NO-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 0
//
// RUN: %clang_cc1 %s -E -dM -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FINITE-MATH-FLAG-UNDEFINED
// CHECK-FINITE-MATH-FLAG-UNDEFINED: #define __FINITE_MATH_ONLY__ 0
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i386 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SYNC_CAS_I386
// CHECK-SYNC_CAS_I386-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i486 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SYNC_CAS_I486
// CHECK-SYNC_CAS_I486: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1
// CHECK-SYNC_CAS_I486: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2
// CHECK-SYNC_CAS_I486: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
// CHECK-SYNC_CAS_I486-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i586 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SYNC_CAS_I586
// CHECK-SYNC_CAS_I586: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1
// CHECK-SYNC_CAS_I586: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2
// CHECK-SYNC_CAS_I586: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
// CHECK-SYNC_CAS_I586: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
