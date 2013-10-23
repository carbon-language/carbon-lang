// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 {
	int a;
	B0() : a(0xf00000B0) {}
};
struct __declspec(align(16)) B1 {
	int a;
	B1() : a(0xf00000B1) {}
};
struct B2 {
	__declspec(align(16)) int a;
	B2() : a(0xf00000B2) {}
};
struct __declspec(align(16)) B3 {
	long long a1;
	int a;
	B3() : a(0xf00000B3), a1(0xf00000B3f00000B3ll) {}
};
struct V {
	char a;
	V() : a(0X11) {}
};
struct __declspec(align(32)) A16 {};
struct V1 : A16 { virtual void f() {} };
struct V2 {
	long long a;
	int a1;
	V2() : a(0xf0000011f0000011ll), a1(0xf0000011) {}
};
struct V3 {
	int a;
	V3() : a(0xf0000022) {}
};
struct __declspec(align(16)) A16X {
};
struct __declspec(align(16)) B0X {
	int a, a1;
	B0X() : a(0xf00000B0), a1(0xf00000B0) {}
};
struct B1X {
	int a;
	B1X() : a(0xf00000B1) {}
};
struct B2X {
	int a;
	B2X() : a(0xf00000B2) {}
};
struct __declspec(align(16)) B3X {
	int a;
	B3X() : a(0xf00000B3) {}
	virtual void g() {}
};
struct B4X : A16X {
	int a, a1;
	B4X() : a(0xf00000B4), a1(0xf00000B4) {}
};
struct B5X : virtual A16X {
	int a, a1;
	B5X() : a(0xf00000B5), a1(0xf00000B5) {}
};
struct B6X {
	int a;
	B6X() : a(0xf00000B6) {}
};

struct A : B1, B0, B2, virtual V {
	int a;
	A() : a(0xf000000A) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   struct B1 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   struct B0 (base)
// CHECK:    4 |     int a
// CHECK:   16 |   struct B2 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   (A vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct V (virtual base)
// CHECK:   64 |     char a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   struct B1 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    4 |   struct B0 (base)
// CHECK-X64:    4 |     int a
// CHECK-X64:   16 |   struct B2 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   (A vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   48 |   struct V (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct B : B2, B0, B1, virtual V {
	int a;
	B() : a(0xf000000B) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   struct B2 (base)
// CHECK:    0 |     int a
// CHECK:   16 |   struct B0 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   struct B1 (base)
// CHECK:   32 |     int a
// CHECK:   36 |   (B vbtable pointer)
// CHECK:   52 |   int a
// CHECK:   64 |   struct V (virtual base)
// CHECK:   64 |     char a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   struct B2 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:   16 |   struct B0 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   struct B1 (base)
// CHECK-X64:   32 |     int a
// CHECK-X64:   40 |   (B vbtable pointer)
// CHECK-X64:   48 |   int a
// CHECK-X64:   64 |   struct V (virtual base)
// CHECK-X64:   64 |     char a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]

struct C : B1, B0, virtual V {
	int a;
	long long a1;
	C() : a(0xf000000C), a1(0xf000000Cf000000Cll) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   struct B1 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   struct B0 (base)
// CHECK:    4 |     int a
// CHECK:    8 |   (C vbtable pointer)
// CHECK:   24 |   int a
// CHECK:   32 |   long long a1
// CHECK:   48 |   struct V (virtual base)
// CHECK:   48 |     char a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   struct B1 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    4 |   struct B0 (base)
// CHECK-X64:    4 |     int a
// CHECK-X64:    8 |   (C vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   long long a1
// CHECK-X64:   32 |   struct V (virtual base)
// CHECK-X64:   32 |     char a
// CHECK-X64:      | [sizeof=48, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct D : B2, B0, virtual V {
	int a;
	D() : a(0xf000000D) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct B2 (base)
// CHECK:    0 |     int a
// CHECK:   16 |   struct B0 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   (D vbtable pointer)
// CHECK:   36 |   int a
// CHECK:   48 |   struct V (virtual base)
// CHECK:   48 |     char a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   struct B2 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:   16 |   struct B0 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   (D vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct V (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct E : B3, B0, virtual V {
	int a;
	E() : a(0xf000000E) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   struct B3 (base)
// CHECK:    0 |     long long a1
// CHECK:    8 |     int a
// CHECK:   16 |   struct B0 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   (E vbtable pointer)
// CHECK:   36 |   int a
// CHECK:   48 |   struct V (virtual base)
// CHECK:   48 |     char a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   struct B3 (base)
// CHECK-X64:    0 |     long long a1
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   struct B0 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   (E vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct V (virtual base)
// CHECK-X64:   48 |     char a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct F : B0, virtual V1 {
	__declspec(align(16)) int a;
	F() : a(0xf000000F) {}
	virtual void f() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   struct B0 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (F vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   92 |   (vtordisp for vbase V1)
// CHECK:   96 |   struct V1 (virtual base)
// CHECK:   96 |     (V1 vftable pointer)
// CHECK:  128 |     struct A16 (base) (empty)
// CHECK:      | [sizeof=128, align=32
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   struct B0 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (F vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   60 |   (vtordisp for vbase V1)
// CHECK-X64:   64 |   struct V1 (virtual base)
// CHECK-X64:   64 |     (V1 vftable pointer)
// CHECK-X64:   96 |     struct A16 (base) (empty)
// CHECK-X64:      | [sizeof=96, align=32
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct G : virtual V2, virtual V3 {
	int a;
	G() : a(0xf0000001) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct G
// CHECK:    0 |   (G vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct V2 (virtual base)
// CHECK:    8 |     long long a
// CHECK:   16 |     int a1
// CHECK:   24 |   struct V3 (virtual base)
// CHECK:   24 |     int a
// CHECK:      | [sizeof=28, align=8
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct G
// CHECK-X64:    0 |   (G vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct V2 (virtual base)
// CHECK-X64:   16 |     long long a
// CHECK-X64:   24 |     int a1
// CHECK-X64:   32 |   struct V3 (virtual base)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct H {
	__declspec(align(16)) int a;
	int b;
	H() : a(0xf0000010), b(0xf0000010) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct H
// CHECK:    0 |   int a
// CHECK:    4 |   int b
// CHECK:      | [sizeof=16, align=16
// CHECK:      |  nvsize=16, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct H
// CHECK-X64:    0 |   int a
// CHECK-X64:    4 |   int b
// CHECK-X64:      | [sizeof=16, align=16
// CHECK-X64:      |  nvsize=16, nvalign=16]

struct I {
	B2 a;
	int b;
	I() : b(0xf0000010) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct I
// CHECK:    0 |   struct B2 a
// CHECK:    0 |     int a
// CHECK:   16 |   int b
// CHECK:      | [sizeof=32, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct I
// CHECK-X64:    0 |   struct B2 a
// CHECK-X64:    0 |     int a
// CHECK-X64:   16 |   int b
// CHECK-X64:      | [sizeof=32, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct AX : B0X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	AX() : a(0xf000000A) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AX
// CHECK:    0 |   (AX vftable pointer)
// CHECK:   16 |   struct B0X (base)
// CHECK:   16 |     int a
// CHECK:   20 |     int a1
// CHECK:   24 |   (AX vbtable pointer)
// CHECK:   40 |   int a
// CHECK:   48 |   struct B2X (virtual base)
// CHECK:   48 |     int a
// CHECK:   52 |   struct B6X (virtual base)
// CHECK:   52 |     int a
// CHECK:   76 |   (vtordisp for vbase B3X)
// CHECK:   80 |   struct B3X (virtual base)
// CHECK:   80 |     (B3X vftable pointer)
// CHECK:   84 |     int a
// CHECK:      | [sizeof=96, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AX
// CHECK-X64:    0 |   (AX vftable pointer)
// CHECK-X64:   16 |   struct B0X (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   20 |     int a1
// CHECK-X64:   24 |   (AX vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B2X (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:   52 |   struct B6X (virtual base)
// CHECK-X64:   52 |     int a
// CHECK-X64:   76 |   (vtordisp for vbase B3X)
// CHECK-X64:   80 |   struct B3X (virtual base)
// CHECK-X64:   80 |     (B3X vftable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct BX : B4X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	BX() : a(0xf000000B) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct BX
// CHECK:    0 |   (BX vftable pointer)
// CHECK:   16 |   struct B4X (base)
// CHECK:   16 |     struct A16X (base) (empty)
// CHECK:   16 |     int a
// CHECK:   20 |     int a1
// CHECK:   32 |   (BX vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct B2X (virtual base)
// CHECK:   64 |     int a
// CHECK:   68 |   struct B6X (virtual base)
// CHECK:   68 |     int a
// CHECK:   92 |   (vtordisp for vbase B3X)
// CHECK:   96 |   struct B3X (virtual base)
// CHECK:   96 |     (B3X vftable pointer)
// CHECK:  100 |     int a
// CHECK:      | [sizeof=112, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct BX
// CHECK-X64:    0 |   (BX vftable pointer)
// CHECK-X64:   16 |   struct B4X (base)
// CHECK-X64:   16 |     struct A16X (base) (empty)
// CHECK-X64:   16 |     int a
// CHECK-X64:   20 |     int a1
// CHECK-X64:   32 |   (BX vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   48 |   struct B2X (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:   52 |   struct B6X (virtual base)
// CHECK-X64:   52 |     int a
// CHECK-X64:   76 |   (vtordisp for vbase B3X)
// CHECK-X64:   80 |   struct B3X (virtual base)
// CHECK-X64:   80 |     (B3X vftable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct CX : B5X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	CX() : a(0xf000000C) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct CX
// CHECK:    0 |   (CX vftable pointer)
// CHECK:   16 |   struct B5X (base)
// CHECK:   16 |     (B5X vbtable pointer)
// CHECK:   20 |     int a
// CHECK:   24 |     int a1
// CHECK:   28 |   int a
// CHECK:   32 |   struct A16X (virtual base) (empty)
// CHECK:   32 |   struct B2X (virtual base)
// CHECK:   32 |     int a
// CHECK:   36 |   struct B6X (virtual base)
// CHECK:   36 |     int a
// CHECK:   60 |   (vtordisp for vbase B3X)
// CHECK:   64 |   struct B3X (virtual base)
// CHECK:   64 |     (B3X vftable pointer)
// CHECK:   68 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct CX
// CHECK-X64:    0 |   (CX vftable pointer)
// CHECK-X64:   16 |   struct B5X (base)
// CHECK-X64:   16 |     (B5X vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:   28 |     int a1
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct A16X (virtual base) (empty)
// CHECK-X64:   48 |   struct B2X (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:   52 |   struct B6X (virtual base)
// CHECK-X64:   52 |     int a
// CHECK-X64:   76 |   (vtordisp for vbase B3X)
// CHECK-X64:   80 |   struct B3X (virtual base)
// CHECK-X64:   80 |     (B3X vftable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct __declspec(align(16)) DX {
	int a;
	DX() : a(0xf000000D) {}
	virtual void f() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct DX
// CHECK:    0 |   (DX vftable pointer)
// CHECK:    4 |   int a
// CHECK:      | [sizeof=16, align=16
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct DX
// CHECK-X64:    0 |   (DX vftable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:      | [sizeof=16, align=16
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
sizeof(AX)+
sizeof(BX)+
sizeof(CX)+
sizeof(DX)];
