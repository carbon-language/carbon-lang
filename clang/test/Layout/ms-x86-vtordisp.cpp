// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 {
	int a;
	B0() : a(0xf00000B0) {}
	virtual void f() { printf("B0"); }
};

struct __declspec(align(16)) B1 {
	int a;
	B1() : a(0xf00000B1) {}
	virtual void f() { printf("B1"); }
};

struct __declspec(align(16)) Align16 {};
struct __declspec(align(32)) Align32 {};
struct VAlign16 : virtual Align16 {};
struct VAlign32 : virtual Align32 {};

struct A : virtual B0, virtual B1 {
	int a;
	A() : a(0xf000000A) {}
	virtual void f() { printf("A"); }
	virtual void g() { printf("A"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct A
// CHECK-NEXT:    0 |   (A vftable pointer)
// CHECK-NEXT:    4 |   (A vbtable pointer)
// CHECK-NEXT:    8 |   int a
// CHECK-NEXT:   16 |   (vtordisp for vbase B0)
// CHECK-NEXT:   20 |   struct B0 (virtual base)
// CHECK-NEXT:   20 |     (B0 vftable pointer)
// CHECK-NEXT:   24 |     int a
// CHECK-NEXT:   44 |   (vtordisp for vbase B1)
// CHECK-NEXT:   48 |   struct B1 (virtual base)
// CHECK-NEXT:   48 |     (B1 vftable pointer)
// CHECK-NEXT:   52 |     int a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=12, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct A
// CHECK-X64-NEXT:    0 |   (A vftable pointer)
// CHECK-X64-NEXT:    8 |   (A vbtable pointer)
// CHECK-X64-NEXT:   16 |   int a
// CHECK-X64-NEXT:   36 |   (vtordisp for vbase B0)
// CHECK-X64-NEXT:   40 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   40 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:   76 |   (vtordisp for vbase B1)
// CHECK-X64-NEXT:   80 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   80 |     (B1 vftable pointer)
// CHECK-X64-NEXT:   88 |     int a
// CHECK-X64-NEXT:      | [sizeof=96, align=16
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=16]

struct C : virtual B0, virtual B1, VAlign32 {
	int a;
	C() : a(0xf000000C) {}
	virtual void f() { printf("C"); }
	virtual void g() { printf("C"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct C
// CHECK-NEXT:    0 |   (C vftable pointer)
// CHECK-NEXT:   32 |   struct VAlign32 (base)
// CHECK-NEXT:   32 |     (VAlign32 vbtable pointer)
// CHECK-NEXT:   36 |   int a
// CHECK-NEXT:   64 |   (vtordisp for vbase B0)
// CHECK-NEXT:   68 |   struct B0 (virtual base)
// CHECK-NEXT:   68 |     (B0 vftable pointer)
// CHECK-NEXT:   72 |     int a
// CHECK-NEXT:  108 |   (vtordisp for vbase B1)
// CHECK-NEXT:  112 |   struct B1 (virtual base)
// CHECK-NEXT:  112 |     (B1 vftable pointer)
// CHECK-NEXT:  116 |     int a
// CHECK-NEXT:  128 |   struct Align32 (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=128, align=32
// CHECK-NEXT:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct C
// CHECK-X64-NEXT:    0 |   (C vftable pointer)
// CHECK-X64-NEXT:   32 |   struct VAlign32 (base)
// CHECK-X64-NEXT:   32 |     (VAlign32 vbtable pointer)
// CHECK-X64-NEXT:   40 |   int a
// CHECK-X64-NEXT:   68 |   (vtordisp for vbase B0)
// CHECK-X64-NEXT:   72 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   72 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   80 |     int a
// CHECK-X64-NEXT:  108 |   (vtordisp for vbase B1)
// CHECK-X64-NEXT:  112 |   struct B1 (virtual base)
// CHECK-X64-NEXT:  112 |     (B1 vftable pointer)
// CHECK-X64-NEXT:  120 |     int a
// CHECK-X64-NEXT:  128 |   struct Align32 (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=128, align=32
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=32]

struct __declspec(align(32)) D : virtual B0, virtual B1  {
	int a;
	D() : a(0xf000000D) {}
	virtual void f() { printf("D"); }
	virtual void g() { printf("D"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct D
// CHECK-NEXT:    0 |   (D vftable pointer)
// CHECK-NEXT:    4 |   (D vbtable pointer)
// CHECK-NEXT:    8 |   int a
// CHECK-NEXT:   32 |   (vtordisp for vbase B0)
// CHECK-NEXT:   36 |   struct B0 (virtual base)
// CHECK-NEXT:   36 |     (B0 vftable pointer)
// CHECK-NEXT:   40 |     int a
// CHECK-NEXT:   76 |   (vtordisp for vbase B1)
// CHECK-NEXT:   80 |   struct B1 (virtual base)
// CHECK-NEXT:   80 |     (B1 vftable pointer)
// CHECK-NEXT:   84 |     int a
// CHECK-NEXT:      | [sizeof=96, align=32
// CHECK-NEXT:      |  nvsize=12, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct D
// CHECK-X64-NEXT:    0 |   (D vftable pointer)
// CHECK-X64-NEXT:    8 |   (D vbtable pointer)
// CHECK-X64-NEXT:   16 |   int a
// CHECK-X64-NEXT:   36 |   (vtordisp for vbase B0)
// CHECK-X64-NEXT:   40 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   40 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:   76 |   (vtordisp for vbase B1)
// CHECK-X64-NEXT:   80 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   80 |     (B1 vftable pointer)
// CHECK-X64-NEXT:   88 |     int a
// CHECK-X64-NEXT:      | [sizeof=96, align=32
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=32]

struct AT {
	virtual ~AT(){}
};
struct CT : virtual AT {
	virtual ~CT();
};
CT::~CT(){}

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct CT
// CHECK-NEXT:    0 |   (CT vbtable pointer)
// CHECK-NEXT:    4 |   struct AT (virtual base)
// CHECK-NEXT:    4 |     (AT vftable pointer)
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct CT
// CHECK-X64-NEXT:    0 |   (CT vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct AT (virtual base)
// CHECK-X64-NEXT:    8 |     (AT vftable pointer)
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct XA {
	XA() { printf("XA"); }
	long long ll;
};
struct XB : XA {
	XB() { printf("XB"); }
	virtual void foo() {}
	int b;
};
struct XC : virtual XB {
	XC() { printf("XC"); }
	virtual void foo() {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct XC
// CHECK-NEXT:    0 |   (XC vbtable pointer)
// CHECK-NEXT:    4 |   (vtordisp for vbase XB)
// CHECK-NEXT:    8 |   struct XB (virtual base)
// CHECK-NEXT:    8 |     (XB vftable pointer)
// CHECK-NEXT:   16 |     struct XA (base)
// CHECK-NEXT:   16 |       long long ll
// CHECK-NEXT:   24 |     int b
// CHECK-NEXT:      | [sizeof=32, align=8
// CHECK-NEXT:      |  nvsize=4, nvalign=8]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct XC
// CHECK-X64-NEXT:    0 |   (XC vbtable pointer)
// CHECK-X64-NEXT:   12 |   (vtordisp for vbase XB)
// CHECK-X64-NEXT:   16 |   struct XB (virtual base)
// CHECK-X64-NEXT:   16 |     (XB vftable pointer)
// CHECK-X64-NEXT:   24 |     struct XA (base)
// CHECK-X64-NEXT:   24 |       long long ll
// CHECK-X64-NEXT:   32 |     int b
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

int a[
sizeof(A)+
sizeof(C)+
sizeof(D)+
sizeof(CT)+
sizeof(XC)];
