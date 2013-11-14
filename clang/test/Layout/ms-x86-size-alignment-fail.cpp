// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { char a; B0() : a(0xB0) {} };
struct __declspec(align(1)) B1 {};

struct A : virtual B0 {};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     char a
// CHECK:      | [sizeof=5, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     char a
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct __declspec(align(1)) B : virtual B0 {};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   (B vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     char a
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   (B vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     char a
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct C : virtual B0 { int a; C() : a(0xC) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base)
// CHECK:    8 |     char a
// CHECK:      | [sizeof=9, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base)
// CHECK-X64:   16 |     char a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct D : virtual B0 { __declspec(align(1)) int a; D() : a(0xD) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   (D vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base)
// CHECK:    8 |     char a
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   (D vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base)
// CHECK-X64:   16 |     char a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct E : virtual B0, virtual B1 {};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   (E vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     char a
// CHECK:    5 |   struct B1 (virtual base) (empty)
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   (E vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     char a
// CHECK-X64:    9 |   struct B1 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct F { char a; virtual ~F(); };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   (F vftable pointer)
// CHECK:    4 |   char a
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   (F vftable pointer)
// CHECK-X64:    8 |   char a
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)];
