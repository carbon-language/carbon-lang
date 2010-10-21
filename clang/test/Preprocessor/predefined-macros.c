// This test verifies that the correct macros are predefined. It currently
// only checks for Microsoft macros.

// RUN: %clang_cc1 %s -E -dM -triple i686-pc-win32 -fms-extensions -fmsc-version=1300 -o - | FileCheck %s


// CHECK: #define _INTEGRAL_MAX_BITS 64
// CHECK: #define _MSC_EXTENSIONS 1
// CHECK: #define _MSC_VER 1300
// CHECK: #define _M_IX86 600
// CHECK: #define _M_IX86_FP
// CHECK: #define _WIN32 1
