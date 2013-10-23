// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
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
// CHECK:    0 | struct A
// CHECK:    0 |   (A vftable pointer)
// CHECK:    4 |   (A vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   16 |   (vtordisp for vbase B0)
// CHECK:   20 |   struct B0 (virtual base)
// CHECK:   20 |     (B0 vftable pointer)
// CHECK:   24 |     int a
// CHECK:   44 |   (vtordisp for vbase B1)
// CHECK:   48 |   struct B1 (virtual base)
// CHECK:   48 |     (B1 vftable pointer)
// CHECK:   52 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   (A vftable pointer)
// CHECK-X64:    8 |   (A vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   36 |   (vtordisp for vbase B0)
// CHECK-X64:   40 |   struct B0 (virtual base)
// CHECK-X64:   40 |     (B0 vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:   76 |   (vtordisp for vbase B1)
// CHECK-X64:   80 |   struct B1 (virtual base)
// CHECK-X64:   80 |     (B1 vftable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct C : virtual B0, virtual B1, VAlign32 {
	int a;
	C() : a(0xf000000C) {}
	virtual void f() { printf("C"); }
	virtual void g() { printf("C"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vftable pointer)
// CHECK:   32 |   struct VAlign32 (base)
// CHECK:   32 |     (VAlign32 vbtable pointer)
// CHECK:   36 |   int a
// CHECK:   64 |   (vtordisp for vbase B0)
// CHECK:   68 |   struct B0 (virtual base)
// CHECK:   68 |     (B0 vftable pointer)
// CHECK:   72 |     int a
// CHECK:  108 |   (vtordisp for vbase B1)
// CHECK:  112 |   struct B1 (virtual base)
// CHECK:  112 |     (B1 vftable pointer)
// CHECK:  116 |     int a
// CHECK:  128 |   struct Align32 (virtual base) (empty)
// CHECK:      | [sizeof=128, align=32
// CHECK:      |  nvsize=64, nvalign=32]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vftable pointer)
// CHECK-X64:   32 |   struct VAlign32 (base)
// CHECK-X64:   32 |     (VAlign32 vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   68 |   (vtordisp for vbase B0)
// CHECK-X64:   72 |   struct B0 (virtual base)
// CHECK-X64:   72 |     (B0 vftable pointer)
// CHECK-X64:   80 |     int a
// CHECK-X64:  108 |   (vtordisp for vbase B1)
// CHECK-X64:  112 |   struct B1 (virtual base)
// CHECK-X64:  112 |     (B1 vftable pointer)
// CHECK-X64:  120 |     int a
// CHECK-X64:  128 |   struct Align32 (virtual base) (empty)
// CHECK-X64:      | [sizeof=128, align=32
// CHECK-X64:      |  nvsize=64, nvalign=32]

struct __declspec(align(32)) D : virtual B0, virtual B1  {
	int a;
	D() : a(0xf000000D) {}
	virtual void f() { printf("D"); }
	virtual void g() { printf("D"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   (D vftable pointer)
// CHECK:    4 |   (D vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   32 |   (vtordisp for vbase B0)
// CHECK:   36 |   struct B0 (virtual base)
// CHECK:   36 |     (B0 vftable pointer)
// CHECK:   40 |     int a
// CHECK:   76 |   (vtordisp for vbase B1)
// CHECK:   80 |   struct B1 (virtual base)
// CHECK:   80 |     (B1 vftable pointer)
// CHECK:   84 |     int a
// CHECK:      | [sizeof=96, align=32
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   (D vftable pointer)
// CHECK-X64:    8 |   (D vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   36 |   (vtordisp for vbase B0)
// CHECK-X64:   40 |   struct B0 (virtual base)
// CHECK-X64:   40 |     (B0 vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:   76 |   (vtordisp for vbase B1)
// CHECK-X64:   80 |   struct B1 (virtual base)
// CHECK-X64:   80 |     (B1 vftable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=32
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AT {
	virtual ~AT(){}
};
struct CT : virtual AT {
	virtual ~CT();
};
CT::~CT(){}

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct CT
// CHECK:    0 |   (CT vbtable pointer)
// CHECK:    4 |   struct AT (virtual base)
// CHECK:    4 |     (AT vftable pointer)
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct CT
// CHECK-X64:    0 |   (CT vbtable pointer)
// CHECK-X64:    8 |   struct AT (virtual base)
// CHECK-X64:    8 |     (AT vftable pointer)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=8, nvalign=8]

int a[
sizeof(A)+
sizeof(C)+
sizeof(D)+
sizeof(CT)];
