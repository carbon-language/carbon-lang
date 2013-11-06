// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

struct B0 { };

struct A : virtual B0 { char a : 1; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vbtable pointer)
// CHECK:    4 |   char a
// CHECK:    9 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=9, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vbtable pointer)
// CHECK-X64:    8 |   char a
// CHECK-X64:   17 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct B : virtual B0 { short a : 1; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   (B vbtable pointer)
// CHECK:    4 |   short a
// CHECK:   10 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=10, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   (B vbtable pointer)
// CHECK-X64:    8 |   short a
// CHECK-X64:   18 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct C : virtual B0 { char a : 1; char : 0; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vbtable pointer)
// CHECK:    4 |   char a
// CHECK:    5 |   char
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vbtable pointer)
// CHECK-X64:    8 |   char a
// CHECK-X64:    9 |   char
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct D : virtual B0 { char a : 1; char b; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   (D vbtable pointer)
// CHECK:    4 |   char a
// CHECK:    5 |   char b
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   (D vbtable pointer)
// CHECK-X64:    8 |   char a
// CHECK-X64:    9 |   char b
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)];
