// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

struct T0 { char c; };
struct T2 : virtual T0 { };
struct T3 { T2 a[1]; char c; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct T3
// CHECK:    0 |   struct T2 [1] a
// CHECK:    5 |   char c
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct T3
// CHECK-X64:    0 |   struct T2 [1] a
// CHECK-X64:   16 |   char c
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

int a[sizeof(T3)];
