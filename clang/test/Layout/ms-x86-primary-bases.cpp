// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) { printf("B0 = %p\n", this); } virtual void f() { printf("B0"); } };
struct B1 { int a; B1() : a(0xf00000B1) { printf("B1 = %p\n", this); } virtual void g() { printf("B1"); } };
struct B2 { int a; B2() : a(0xf00000B2) { printf("B1 = %p\n", this); } };
struct B0X { int a; B0X() : a(0xf00000B0) {} };
struct B1X { int a; B1X() : a(0xf00000B1) {} virtual void f() { printf("B0"); } };
struct B2X : virtual B1X { int a; B2X() : a(0xf00000B2) {} };

struct A : virtual B0 {
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     (B0 vftable pointer)
// CHECK:    8 |     int a
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     (B0 vftable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct B : virtual B0 {
	virtual void f() { printf("B"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   (B vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     (B0 vftable pointer)
// CHECK:    8 |     int a
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   (B vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     (B0 vftable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct C : virtual B0 {
	virtual void g() { printf("A"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vftable pointer)
// CHECK:    4 |   (C vbtable pointer)
// CHECK:    8 |   struct B0 (virtual base)
// CHECK:    8 |     (B0 vftable pointer)
// CHECK:   12 |     int a
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vftable pointer)
// CHECK-X64:    8 |   (C vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base)
// CHECK-X64:   16 |     (B0 vftable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:      | [sizeof=32, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct D : virtual B2, virtual B0 {
	virtual void f() { printf("D"); }
	virtual void g() { printf("D"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   (D vftable pointer)
// CHECK:    4 |   (D vbtable pointer)
// CHECK:    8 |   struct B2 (virtual base)
// CHECK:    8 |     int a
// CHECK:   12 |   struct B0 (virtual base)
// CHECK:   12 |     (B0 vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   (D vftable pointer)
// CHECK-X64:    8 |   (D vbtable pointer)
// CHECK-X64:   16 |   struct B2 (virtual base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct B0 (virtual base)
// CHECK-X64:   24 |     (B0 vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct E : B0, virtual B1 {
	virtual void f() { printf("E"); }
	virtual void g() { printf("E"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   struct B0 (primary base)
// CHECK:    0 |     (B0 vftable pointer)
// CHECK:    4 |     int a
// CHECK:    8 |   (E vbtable pointer)
// CHECK:   12 |   struct B1 (virtual base)
// CHECK:   12 |     (B1 vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   struct B0 (primary base)
// CHECK-X64:    0 |     (B0 vftable pointer)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (E vbtable pointer)
// CHECK-X64:   24 |   struct B1 (virtual base)
// CHECK-X64:   24 |     (B1 vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct F : virtual B0, virtual B1 {
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   (F vbtable pointer)
// CHECK:    4 |   struct B0 (virtual base)
// CHECK:    4 |     (B0 vftable pointer)
// CHECK:    8 |     int a
// CHECK:   12 |   struct B1 (virtual base)
// CHECK:   12 |     (B1 vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   (F vbtable pointer)
// CHECK-X64:    8 |   struct B0 (virtual base)
// CHECK-X64:    8 |     (B0 vftable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct B1 (virtual base)
// CHECK-X64:   24 |     (B1 vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

struct AX : B0X, B1X { int a; AX() : a(0xf000000A) {} virtual void f() { printf("A"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AX
// CHECK:    8 |   struct B0X (base)
// CHECK:    8 |     int a
// CHECK:    0 |   struct B1X (primary base)
// CHECK:    0 |     (B1X vftable pointer)
// CHECK:    4 |     int a
// CHECK:   12 |   int a
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AX
// CHECK-X64:   16 |   struct B0X (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:    0 |   struct B1X (primary base)
// CHECK-X64:    0 |     (B1X vftable pointer)
// CHECK-X64:    8 |     int a
// CHECK-X64:   20 |   int a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct BX : B0X, B1X { int a; BX() : a(0xf000000B) {} virtual void g() { printf("B"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct BX
// CHECK:    8 |   struct B0X (base)
// CHECK:    8 |     int a
// CHECK:    0 |   struct B1X (primary base)
// CHECK:    0 |     (B1X vftable pointer)
// CHECK:    4 |     int a
// CHECK:   12 |   int a
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-x64: *** Dumping AST Record Layout
// CHECK-x64:    0 | struct BX
// CHECK-x64:   16 |   struct B0X (base)
// CHECK-x64:   16 |     int a
// CHECK-x64:    0 |   struct B1X (primary base)
// CHECK-x64:    0 |     (B1X vftable pointer)
// CHECK-x64:    8 |     int a
// CHECK-x64:   24 |   int a
// CHECK-x64:      | [sizeof=24, align=8
// CHECK-x64:      |  nvsize=24, nvalign=8]

struct CX : B0X, B2X { int a; CX() : a(0xf000000C) {} virtual void g() { printf("C"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct CX
// CHECK:    0 |   (CX vftable pointer)
// CHECK:    4 |   struct B0X (base)
// CHECK:    4 |     int a
// CHECK:    8 |   struct B2X (base)
// CHECK:    8 |     (B2X vbtable pointer)
// CHECK:   12 |     int a
// CHECK:   16 |   int a
// CHECK:   20 |   struct B1X (virtual base)
// CHECK:   20 |     (B1X vftable pointer)
// CHECK:   24 |     int a
// CHECK:      | [sizeof=28, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct CX
// CHECK-X64:    0 |   (CX vftable pointer)
// CHECK-X64:    8 |   struct B0X (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   struct B2X (base)
// CHECK-X64:   16 |     (B2X vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   40 |   struct B1X (virtual base)
// CHECK-X64:   40 |     (B1X vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=56, align=8
// CHECK-X64:      |  nvsize=40, nvalign=8]

struct DX : virtual B1X { int a; DX() : a(0xf000000D) {} virtual void f() { printf("D"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct DX
// CHECK:    0 |   (DX vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   (vtordisp for vbase B1X)
// CHECK:   12 |   struct B1X (virtual base)
// CHECK:   12 |     (B1X vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct DX
// CHECK-X64:    0 |   (DX vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   20 |   (vtordisp for vbase B1X)
// CHECK-X64:   24 |   struct B1X (virtual base)
// CHECK-X64:   24 |     (B1X vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct EX : virtual B1X { int a; EX() : a(0xf000000E) {} virtual void g() { printf("E"); } };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct EX
// CHECK:    0 |   (EX vftable pointer)
// CHECK:    4 |   (EX vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   12 |   struct B1X (virtual base)
// CHECK:   12 |     (B1X vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct EX
// CHECK-X64:    0 |   (EX vftable pointer)
// CHECK-X64:    8 |   (EX vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B1X (virtual base)
// CHECK-X64:   24 |     (B1X vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct FX : virtual B1X { int a; FX() : a(0xf000000F) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct FX
// CHECK:    0 |   (FX vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B1X (virtual base)
// CHECK:    8 |     (B1X vftable pointer)
// CHECK:   12 |     int a
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct FX
// CHECK-X64:    0 |   (FX vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B1X (virtual base)
// CHECK-X64:   16 |     (B1X vftable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:      | [sizeof=32, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(AX)+
sizeof(BX)+
sizeof(CX)+
sizeof(DX)+
sizeof(EX)+
sizeof(FX)];
