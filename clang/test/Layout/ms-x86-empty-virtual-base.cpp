// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct __declspec(align(8)) B0 { B0() {printf("B0 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct __declspec(align(8)) B1 { B1() {printf("B1 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct __declspec(align(8)) B2 { B2() {printf("B2 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct __declspec(align(8)) B3 { B3() {printf("B3 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct __declspec(align(8)) B4 { B4() {printf("B4 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };

struct C0 { int a; C0() : a(0xf00000C0) {printf("C0 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct C1 { int a; C1() : a(0xf00000C1) {printf("C1 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct C2 { int a; C2() : a(0xf00000C2) {printf("C2 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct C3 { int a; C3() : a(0xf00000C3) {printf("C3 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct C4 { int a; C4() : a(0xf00000C4) {printf("C4 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };

struct __declspec(align(16)) D0 { D0() {printf("D0 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} virtual void f() {} };
struct D1 { D1() {printf("D1 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };
struct D2 { int a[8]; D2() {printf("D2 : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);} };

struct A : virtual B0 {
	int a;
	A() : a(0xf000000A) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=8, align=8
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct B : virtual B0 {
	B0 b0;
	int a;
	B() : a(0xf000000B) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   (B vbtable pointer)
// CHECK:    8 |   struct B0 b0 (empty)
// CHECK:      |   [sizeof=8, align=8
// CHECK:      |    nvsize=0, nvalign=1]
// CHECK:   16 |   int a
// CHECK:   24 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=24, align=8
// CHECK:      |  nvsize=24, nvalign=8]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   (B vbtable pointer)
// CHECK-X64:    8 |   struct B0 b0 (empty)
// CHECK-X64:      |   [sizeof=8, align=8
// CHECK-X64:      |    nvsize=0, nvalign=1]
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct C : virtual B0, virtual B1, virtual B2, virtual B3, virtual B4 {
	int a;
	C() : a(0xf000000C) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:   16 |   struct B1 (virtual base) (empty)
// CHECK:   24 |   struct B2 (virtual base) (empty)
// CHECK:   32 |   struct B3 (virtual base) (empty)
// CHECK:   40 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=40, align=8
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:   24 |   struct B1 (virtual base) (empty)
// CHECK-X64:   32 |   struct B2 (virtual base) (empty)
// CHECK-X64:   40 |   struct B3 (virtual base) (empty)
// CHECK-X64:   48 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct D {
	B0 b0;
	C0 c0;
	C1 c1;
	C2 c2;
	B1 b1;
	int a;
	D() : a(0xf000000D) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct B0 b0 (empty)
// CHECK:    8 |   struct C0 c0
// CHECK:    8 |     int a
// CHECK:   12 |   struct C1 c1
// CHECK:   12 |     int a
// CHECK:   16 |   struct C2 c2
// CHECK:   16 |     int a
// CHECK:   24 |   struct B1 b1 (empty)
// CHECK:   32 |   int a
// CHECK:      | [sizeof=40, align=8
// CHECK:      |  nvsize=40, nvalign=8]
// CHECK-64: *** Dumping AST Record Layout
// CHECK-64:    0 | struct D
// CHECK-64:    0 |   struct B0 b0 (empty)
// CHECK-64:    8 |   struct C0 c0
// CHECK-64:    8 |     int a
// CHECK-64:   12 |   struct C1 c1
// CHECK-64:   12 |     int a
// CHECK-64:   16 |   struct C2 c2
// CHECK-64:   16 |     int a
// CHECK-64:   24 |   struct B1 b1 (empty)
// CHECK-64:   32 |   int a
// CHECK-64:      | [sizeof=40, align=8
// CHECK-64:      |  nvsize=40, nvalign=8]

struct E : virtual B0, virtual C0, virtual C1, virtual C2, virtual B1 {
	int a;
	E() : a(0xf000000E) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   (E vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:    8 |   struct C0 (virtual base)
// CHECK:    8 |     int a
// CHECK:   12 |   struct C1 (virtual base)
// CHECK:   12 |     int a
// CHECK:   16 |   struct C2 (virtual base)
// CHECK:   16 |     int a
// CHECK:   24 |   struct B1 (virtual base) (empty)
// CHECK:      | [sizeof=24, align=8
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   (E vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:   16 |   struct C0 (virtual base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   20 |   struct C1 (virtual base)
// CHECK-X64:   20 |     int a
// CHECK-X64:   24 |   struct C2 (virtual base)
// CHECK-X64:   24 |     int a
// CHECK-X64:   32 |   struct B1 (virtual base) (empty)
// CHECK-X64:      | [sizeof=32, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct F : virtual C0, virtual B0, virtual B1, virtual C1 {
	int a;
	F() : a(0xf000000F) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   (F vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct C0 (virtual base)
// CHECK:    8 |     int a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:   24 |   struct B1 (virtual base) (empty)
// CHECK:   24 |   struct C1 (virtual base)
// CHECK:   24 |     int a
// CHECK:      | [sizeof=32, align=8
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   (F vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct C0 (virtual base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:   32 |   struct B1 (virtual base) (empty)
// CHECK-X64:   32 |   struct C1 (virtual base)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct G : virtual C0, virtual B0, virtual B1, D0, virtual C1 {
	int a;
	G() : a(0xf0000010) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
	virtual void f() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct G
// CHECK:    0 |   struct D0 (primary base)
// CHECK:    0 |     (D0 vftable pointer)
// CHECK:    4 |   (G vbtable pointer)
// CHECK:   20 |   int a
// CHECK:   32 |   struct C0 (virtual base)
// CHECK:   32 |     int a
// CHECK:   40 |   struct B0 (virtual base) (empty)
// CHECK:   56 |   struct B1 (virtual base) (empty)
// CHECK:   56 |   struct C1 (virtual base)
// CHECK:   56 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct G
// CHECK-X64:    0 |   struct D0 (primary base)
// CHECK-X64:    0 |     (D0 vftable pointer)
// CHECK-X64:    8 |   (G vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   32 |   struct C0 (virtual base)
// CHECK-X64:   32 |     int a
// CHECK-X64:   40 |   struct B0 (virtual base) (empty)
// CHECK-X64:   56 |   struct B1 (virtual base) (empty)
// CHECK-X64:   56 |   struct C1 (virtual base)
// CHECK-X64:   56 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct H : virtual C0, virtual B0, virtual B1, virtual D0, virtual C1 {
	int a;
	H() : a(0xf0000011) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
	virtual void f() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct H
// CHECK:    0 |   (H vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct C0 (virtual base)
// CHECK:    8 |     int a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:   24 |   struct B1 (virtual base) (empty)
// CHECK:   44 |   (vtordisp for vbase D0)
// CHECK:   48 |   struct D0 (virtual base)
// CHECK:   48 |     (D0 vftable pointer)
// CHECK:   52 |   struct C1 (virtual base)
// CHECK:   52 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct H
// CHECK-X64:    0 |   (H vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct C0 (virtual base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:   40 |   struct B1 (virtual base) (empty)
// CHECK-X64:   60 |   (vtordisp for vbase D0)
// CHECK-X64:   64 |   struct D0 (virtual base)
// CHECK-X64:   64 |     (D0 vftable pointer)
// CHECK-X64:   72 |   struct C1 (virtual base)
// CHECK-X64:   72 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct I : virtual B0, virtual B1, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	I() : a(0xf0000012) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct I
// CHECK:    0 |   (I vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B0 (virtual base) (empty)
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct I
// CHECK-X64:    0 |   (I vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B0 (virtual base) (empty)
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct __declspec(align(32)) J : virtual B0, virtual B1, virtual B2, virtual B3, virtual B4 {
	int a;
	J() : a(0xf0000012) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct J
// CHECK:    0 |   (J vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:   40 |   struct B1 (virtual base) (empty)
// CHECK:   72 |   struct B2 (virtual base) (empty)
// CHECK:  104 |   struct B3 (virtual base) (empty)
// CHECK:  136 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=160, align=32
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct J
// CHECK-X64:    0 |   (J vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:   40 |   struct B1 (virtual base) (empty)
// CHECK-X64:   72 |   struct B2 (virtual base) (empty)
// CHECK-X64:  104 |   struct B3 (virtual base) (empty)
// CHECK-X64:  136 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=160, align=32
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct K : virtual D1, virtual B1, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	K() : a(0xf0000013) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct K
// CHECK:    0 |   (K vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct D1 (virtual base) (empty)
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct K
// CHECK-X64:    0 |   (K vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct D1 (virtual base) (empty)
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct L : virtual B1, virtual D1, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	L() : a(0xf0000014) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct L
// CHECK:    0 |   (L vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B1 (virtual base) (empty)
// CHECK:   68 |   struct D1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct L
// CHECK-X64:    0 |   (L vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B1 (virtual base) (empty)
// CHECK-X64:   68 |   struct D1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct M : virtual B1, virtual B2, virtual D1, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	M() : a(0xf0000015) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct M
// CHECK:    0 |   (M vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B1 (virtual base) (empty)
// CHECK:   72 |   struct B2 (virtual base) (empty)
// CHECK:  100 |   struct D1 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct M
// CHECK-X64:    0 |   (M vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B1 (virtual base) (empty)
// CHECK-X64:   72 |   struct B2 (virtual base) (empty)
// CHECK-X64:  100 |   struct D1 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct N : virtual C0, virtual B1, virtual D1, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	N() : a(0xf0000016) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct N
// CHECK:    0 |   (N vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct C0 (virtual base)
// CHECK:   64 |     int a
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:  100 |   struct D1 (virtual base) (empty)
// CHECK:  136 |   struct B2 (virtual base) (empty)
// CHECK:  168 |   struct B3 (virtual base) (empty)
// CHECK:  200 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=224, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct N
// CHECK-X64:    0 |   (N vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct C0 (virtual base)
// CHECK-X64:   64 |     int a
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:  100 |   struct D1 (virtual base) (empty)
// CHECK-X64:  136 |   struct B2 (virtual base) (empty)
// CHECK-X64:  168 |   struct B3 (virtual base) (empty)
// CHECK-X64:  200 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=224, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct O : virtual C0, virtual B1, virtual B2, virtual D1, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	O() : a(0xf0000017) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct O
// CHECK:    0 |   (O vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct C0 (virtual base)
// CHECK:   64 |     int a
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  132 |   struct D1 (virtual base) (empty)
// CHECK:  168 |   struct B3 (virtual base) (empty)
// CHECK:  200 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=224, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct O
// CHECK-X64:    0 |   (O vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct C0 (virtual base)
// CHECK-X64:   64 |     int a
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  132 |   struct D1 (virtual base) (empty)
// CHECK-X64:  168 |   struct B3 (virtual base) (empty)
// CHECK-X64:  200 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=224, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct P : virtual B1, virtual C0, virtual D1, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	P() : a(0xf0000018) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct P
// CHECK:    0 |   (P vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B1 (virtual base) (empty)
// CHECK:   64 |   struct C0 (virtual base)
// CHECK:   64 |     int a
// CHECK:   68 |   struct D1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct P
// CHECK-X64:    0 |   (P vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B1 (virtual base) (empty)
// CHECK-X64:   64 |   struct C0 (virtual base)
// CHECK-X64:   64 |     int a
// CHECK-X64:   68 |   struct D1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct Q : virtual B1, virtual C0, virtual B2, virtual D1, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	Q() : a(0xf0000019) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct Q
// CHECK:    0 |   (Q vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B1 (virtual base) (empty)
// CHECK:   64 |   struct C0 (virtual base)
// CHECK:   64 |     int a
// CHECK:   72 |   struct B2 (virtual base) (empty)
// CHECK:  100 |   struct D1 (virtual base) (empty)
// CHECK:  136 |   struct B3 (virtual base) (empty)
// CHECK:  168 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=192, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct Q
// CHECK-X64:    0 |   (Q vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B1 (virtual base) (empty)
// CHECK-X64:   64 |   struct C0 (virtual base)
// CHECK-X64:   64 |     int a
// CHECK-X64:   72 |   struct B2 (virtual base) (empty)
// CHECK-X64:  100 |   struct D1 (virtual base) (empty)
// CHECK-X64:  136 |   struct B3 (virtual base) (empty)
// CHECK-X64:  168 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=192, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct R : virtual B0, virtual B1, virtual B2, virtual C0, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	R() : a(0xf0000020) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct R
// CHECK:    0 |   (R vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B0 (virtual base) (empty)
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:  104 |   struct B2 (virtual base) (empty)
// CHECK:  104 |   struct C0 (virtual base)
// CHECK:  104 |     int a
// CHECK:  112 |   struct B3 (virtual base) (empty)
// CHECK:  136 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=160, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct R
// CHECK-X64:    0 |   (R vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B0 (virtual base) (empty)
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:  104 |   struct B2 (virtual base) (empty)
// CHECK-X64:  104 |   struct C0 (virtual base)
// CHECK-X64:  104 |     int a
// CHECK-X64:  112 |   struct B3 (virtual base) (empty)
// CHECK-X64:  136 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=160, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct S : virtual B0, virtual B1, virtual C0, virtual B2, virtual B3, virtual B4 {
	__declspec(align(32)) int a;
	S() : a(0xf0000021) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct S
// CHECK:    0 |   (S vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   64 |   struct B0 (virtual base) (empty)
// CHECK:   72 |   struct B1 (virtual base) (empty)
// CHECK:   72 |   struct C0 (virtual base)
// CHECK:   72 |     int a
// CHECK:   80 |   struct B2 (virtual base) (empty)
// CHECK:  104 |   struct B3 (virtual base) (empty)
// CHECK:  136 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=160, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct S
// CHECK-X64:    0 |   (S vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   64 |   struct B0 (virtual base) (empty)
// CHECK-X64:   72 |   struct B1 (virtual base) (empty)
// CHECK-X64:   72 |   struct C0 (virtual base)
// CHECK-X64:   72 |     int a
// CHECK-X64:   80 |   struct B2 (virtual base) (empty)
// CHECK-X64:  104 |   struct B3 (virtual base) (empty)
// CHECK-X64:  136 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=160, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct T : virtual B0, virtual B1, virtual C0, virtual D2, virtual B2, virtual B3, virtual B4 {
	__declspec(align(16)) int a;
	T() : a(0xf0000022) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct T
// CHECK:    0 |   (T vbtable pointer)
// CHECK:   16 |   int a
// CHECK:   32 |   struct B0 (virtual base) (empty)
// CHECK:   40 |   struct B1 (virtual base) (empty)
// CHECK:   40 |   struct C0 (virtual base)
// CHECK:   40 |     int a
// CHECK:   44 |   struct D2 (virtual base)
// CHECK:   44 |     int [8] a
// CHECK:   80 |   struct B2 (virtual base) (empty)
// CHECK:   88 |   struct B3 (virtual base) (empty)
// CHECK:  104 |   struct B4 (virtual base) (empty)
// CHECK:      | [sizeof=112, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct T
// CHECK-X64:    0 |   (T vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   32 |   struct B0 (virtual base) (empty)
// CHECK-X64:   40 |   struct B1 (virtual base) (empty)
// CHECK-X64:   40 |   struct C0 (virtual base)
// CHECK-X64:   40 |     int a
// CHECK-X64:   44 |   struct D2 (virtual base)
// CHECK-X64:   44 |     int [8] a
// CHECK-X64:   80 |   struct B2 (virtual base) (empty)
// CHECK-X64:   88 |   struct B3 (virtual base) (empty)
// CHECK-X64:  104 |   struct B4 (virtual base) (empty)
// CHECK-X64:      | [sizeof=112, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct __declspec(align(32)) U : virtual B0, virtual B1 {
	int a;
	U() : a(0xf0000023) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct U
// CHECK:    0 |   (U vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:   40 |   struct B1 (virtual base) (empty)
// CHECK:      | [sizeof=64, align=32
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct U
// CHECK-X64:    0 |   (U vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:   40 |   struct B1 (virtual base) (empty)
// CHECK-X64:      | [sizeof=64, align=32
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct __declspec(align(32)) V : virtual D1 {
	int a;
	V() : a(0xf0000024) {printf("X : %3d\n", ((int)(__SIZE_TYPE__)this)&0xfff);}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct V
// CHECK:    0 |   (V vbtable pointer)
// CHECK:    4 |   int a
// CHECK:    8 |   struct D1 (virtual base) (empty)
// CHECK:      | [sizeof=32, align=32
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct V
// CHECK-X64:    0 |   (V vbtable pointer)
// CHECK-X64:    8 |   int a
// CHECK-X64:   16 |   struct D1 (virtual base) (empty)
// CHECK-X64:      | [sizeof=32, align=32
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
sizeof(M)+
sizeof(N)+
sizeof(O)+
sizeof(P)+
sizeof(Q)+
sizeof(R)+
sizeof(S)+
sizeof(T)+
sizeof(U)+
sizeof(V)];