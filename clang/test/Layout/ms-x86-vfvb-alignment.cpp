// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) {} };
struct B1 { char a; B1() : a(0xB1) {} };
struct B2 : virtual B1 { int a; B2() : a(0xf00000B2) {} };
struct B3 { __declspec(align(16)) int a; B3() : a(0xf00000B3) {} };
struct B4 : virtual B3 { int a; B4() : a(0xf00000B4) {} };
struct B5 { __declspec(align(32)) int a; B5() : a(0xf00000B5) {} };
struct B6 { int a; B6() : a(0xf00000B6) {} virtual void f() { printf("B6"); } };

struct A : B0, virtual B1 { __declspec(align(16)) int a; A() : a(0xf000000A) {} virtual void f() { printf("A"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vftable pointer)
// CHECK:   16 |   struct B0 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   (A vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct B1 (virtual base)
// CHECK:   64 |     char a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vftable pointer)
// CHECK-X64:    8 |   struct B0 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (A vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct B : A, B2 { int a; B() : a(0xf000000B) {} virtual void f() { printf("B"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   struct A (primary base)
// CHECK:    0 |     (A vftable pointer)
// CHECK:   16 |     struct B0 (base)
// CHECK:   16 |       int a
// CHECK:   20 |     (A vbtable pointer)
// CHECK:   48 |     int a
// CHECK:   64 |   struct B2 (base)
// CHECK:   64 |     (B2 vbtable pointer)
// CHECK:   68 |     int a
// CHECK:   72 |   int a
// CHECK:   80 |   struct B1 (virtual base)
// CHECK:   80 |     char a
// CHECK:      | [sizeof=96, align=16
// CHECK:      |  nvsize=80, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   struct A (primary base)
// CHECK-X64:    0 |     (A vftable pointer)
// CHECK-X64:    8 |     struct B0 (base)
// CHECK-X64:    8 |       int a
// CHECK-X64:   16 |     (A vbtable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:   48 |   struct B2 (base)
// CHECK-X64:   48 |     (B2 vbtable pointer)
// CHECK-X64:   56 |     int a
// CHECK-X64:   64 |   int a
// CHECK-X64:   80 |   struct B1 (virtual base)
// CHECK-X64:   80 |     char a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=80, nvalign=16]

struct C : B4 { int a; C() : a(0xf000000C) {} virtual void f() { printf("C"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vftable pointer)
// CHECK:   16 |   struct B4 (base)
// CHECK:   16 |     (B4 vbtable pointer)
// CHECK:   20 |     int a
// CHECK:   24 |   int a
// CHECK:   32 |   struct B3 (virtual base)
// CHECK:   32 |     int a
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vftable pointer)
// CHECK-X64:   16 |   struct B4 (base)
// CHECK-X64:   16 |     (B4 vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B3 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct D : C { int a; D() : a(0xf000000D) {} virtual void f() { printf("D"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct C (primary base)
// CHECK:    0 |     (C vftable pointer)
// CHECK:   16 |     struct B4 (base)
// CHECK:   16 |       (B4 vbtable pointer)
// CHECK:   20 |       int a
// CHECK:   24 |     int a
// CHECK:   32 |   int a
// CHECK:   48 |   struct B3 (virtual base)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   struct C (primary base)
// CHECK-X64:    0 |     (C vftable pointer)
// CHECK-X64:   16 |     struct B4 (base)
// CHECK-X64:   16 |       (B4 vbtable pointer)
// CHECK-X64:   24 |       int a
// CHECK-X64:   32 |     int a
// CHECK-X64:   48 |   int a
// CHECK-X64:   64 |   struct B3 (virtual base)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]

struct E : virtual C { int a; E() : a(0xf000000E) {} virtual void f() { printf("E"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   (E vbtable pointer)
// CHECK:    4 |   int a
// CHECK:   16 |   struct B3 (virtual base)
// CHECK:   16 |     int a
// CHECK:   44 |   (vtordisp for vbase C)
// CHECK:   48 |   struct C (virtual base)
// CHECK:   48 |     (C vftable pointer)
// CHECK:   64 |     struct B4 (base)
// CHECK:   64 |       (B4 vbtable pointer)
// CHECK:   68 |       int a
// CHECK:   72 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   (E vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B3 (virtual base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   44 |   (vtordisp for vbase C)
// CHECK-X64:   48 |   struct C (virtual base)
// CHECK-X64:   48 |     (C vftable pointer)
// CHECK-X64:   64 |     struct B4 (base)
// CHECK-X64:   64 |       (B4 vbtable pointer)
// CHECK-X64:   72 |       int a
// CHECK-X64:   80 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct F : B3, virtual B0 { int a; F() : a(0xf000000F) {} virtual void f() { printf("F"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   (F vftable pointer)
// CHECK:   16 |   struct B3 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   (F vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct B0 (virtual base)
// CHECK:   64 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   (F vftable pointer)
// CHECK-X64:   16 |   struct B3 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   (F vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   48 |   struct B0 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct G : B2, B6, virtual B1 { int a; G() : a(0xf0000010) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct G
// CHECK:    8 |   struct B2 (base)
// CHECK:    8 |     (B2 vbtable pointer)
// CHECK:   12 |     int a
// CHECK:    0 |   struct B6 (primary base)
// CHECK:    0 |     (B6 vftable pointer)
// CHECK:    4 |     int a
// CHECK:   16 |   int a
// CHECK:   20 |   struct B1 (virtual base)
// CHECK:   20 |     char a
// CHECK:      | [sizeof=21, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct G
// CHECK-X64:   16 |   struct B2 (base)
// CHECK-X64:   16 |     (B2 vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:    0 |   struct B6 (primary base)
// CHECK-X64:    0 |     (B6 vftable pointer)
// CHECK-X64:    8 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   40 |   struct B1 (virtual base)
// CHECK-X64:   40 |     char a
// CHECK-X64:      | [sizeof=48, align=8
// CHECK-X64:      |  nvsize=40, nvalign=8]

struct H : B6, B2, virtual B1 { int a; H() : a(0xf0000011) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct H
// CHECK:    0 |   struct B6 (primary base)
// CHECK:    0 |     (B6 vftable pointer)
// CHECK:    4 |     int a
// CHECK:    8 |   struct B2 (base)
// CHECK:    8 |     (B2 vbtable pointer)
// CHECK:   12 |     int a
// CHECK:   16 |   int a
// CHECK:   20 |   struct B1 (virtual base)
// CHECK:   20 |     char a
// CHECK:      | [sizeof=21, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct H
// CHECK-X64:    0 |   struct B6 (primary base)
// CHECK-X64:    0 |     (B6 vftable pointer)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   struct B2 (base)
// CHECK-X64:   16 |     (B2 vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   40 |   struct B1 (virtual base)
// CHECK-X64:   40 |     char a
// CHECK-X64:      | [sizeof=48, align=8
// CHECK-X64:      |  nvsize=40, nvalign=8]

struct I : B0, virtual B1 { int a; int a1; __declspec(align(16)) int a2; I() : a(0xf0000011), a1(0xf0000011), a2(0xf0000011) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct I
// CHECK:    0 |   struct B0 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (I vbtable pointer)
// CHECK:   20 |   int a
// CHECK:   24 |   int a1
// CHECK:   32 |   int a2
// CHECK:   48 |   struct B1 (virtual base)
// CHECK:   48 |     char a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct I
// CHECK-X64:    0 |   struct B0 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (I vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   20 |   int a1
// CHECK-X64:   32 |   int a2
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct J : B0, B3, virtual B1 { int a; int a1; J() : a(0xf0000012), a1(0xf0000012) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct J
// CHECK:    0 |   struct B0 (base)
// CHECK:    0 |     int a
// CHECK:   16 |   struct B3 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   (J vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   52 |   int a1
// CHECK:   64 |   struct B1 (virtual base)
// CHECK:   64 |     char a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct J
// CHECK-X64:    0 |   struct B0 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:   16 |   struct B3 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   (J vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   44 |   int a1
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct K { int a; K() : a(0xf0000013) {} virtual void f() { printf("K"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct K
// CHECK:    0 |   (K vftable pointer)
// CHECK:    4 |   int a
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct K
// CHECK-X64:    0 |   (K vftable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct L : virtual K { int a; L() : a(0xf0000014) {} virtual void g() { printf("L"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct L
// CHECK:    0 |   (L vftable pointer)
// CHECK:    4 |   (L vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   12 |   struct K (virtual base)
// CHECK:   12 |     (K vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct L
// CHECK-X64:    0 |   (L vftable pointer)
// CHECK-X64:    8 |   (L vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct K (virtual base)
// CHECK-X64:   24 |     (K vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct M : virtual K { int a; M() : a(0xf0000015) {} virtual void f() { printf("M"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct M
// CHECK:    0 |   (M vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   (vtordisp for vbase K)
// CHECK:   12 |   struct K (virtual base)
// CHECK:   12 |     (K vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct M
// CHECK-X64:    0 |   (M vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   20 |   (vtordisp for vbase K)
// CHECK-X64:   24 |   struct K (virtual base)
// CHECK-X64:   24 |     (K vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(G)+
sizeof(H)+
sizeof(I)+
sizeof(J)+
sizeof(K)+
sizeof(L)+
sizeof(M)];
