// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) { printf("B0 = %p\n", this); } };
struct B1 { int a; B1() : a(0xf00000B1) { printf("B1 = %p\n", this); } };
struct B2 { B2() { printf("B2 = %p\n", this); } virtual void g() { printf("B2"); } };
struct B3 : virtual B1 { B3() { printf("B3 = %p\n", this); } };
struct B4 : virtual B1 { B4() { printf("B4 = %p\n", this); } virtual void g() { printf("B4"); } };

struct A : B0, virtual B1 {
	__declspec(align(16)) int a;
	A() : a(0xf000000A) { printf(" A = %p\n\n", this); }
	virtual void f() { printf("A"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   (A vftable pointer)
// CHECK:   16 |   struct B0 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   (A vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct B1 (virtual base)
// CHECK:   64 |     int a
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
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct B : B2, B0, virtual B1 {
	__declspec(align(16)) int a;
	B() : a(0xf000000B) { printf(" B = %p\n\n", this); }
	virtual void f() { printf("B"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   struct B2 (primary base)
// CHECK:    0 |     (B2 vftable pointer)
// CHECK:    4 |   struct B0 (base)
// CHECK:    4 |     int a
// CHECK:    8 |   (B vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   48 |   struct B1 (virtual base)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   struct B2 (primary base)
// CHECK-X64:    0 |     (B2 vftable pointer)
// CHECK-X64:    8 |   struct B0 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (B vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct C : B3, B0, virtual B1 {
	__declspec(align(16)) int a;
	C() : a(0xf000000C) { printf(" C = %p\n\n", this); }
	virtual void f() { printf("C"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   (C vftable pointer)
// CHECK:   16 |   struct B3 (base)
// CHECK:   16 |     (B3 vbtable pointer)
// CHECK:   20 |   struct B0 (base)
// CHECK:   20 |     int a
// CHECK:   32 |   int a
// CHECK:   48 |   struct B1 (virtual base)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   (C vftable pointer)
// CHECK-X64:    8 |   struct B3 (base)
// CHECK-X64:    8 |     (B3 vbtable pointer)
// CHECK-X64:   16 |   struct B0 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct D : B4, B0, virtual B1 {
	__declspec(align(16)) int a;
	D() : a(0xf000000D) { printf(" D = %p\n\n", this); }
	virtual void f() { printf("D"); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct B4 (primary base)
// CHECK:    0 |     (B4 vftable pointer)
// CHECK:    4 |     (B4 vbtable pointer)
// CHECK:    8 |   struct B0 (base)
// CHECK:    8 |     int a
// CHECK:   16 |   int a
// CHECK:   32 |   struct B1 (virtual base)
// CHECK:   32 |     int a
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   struct B4 (primary base)
// CHECK-X64:    0 |     (B4 vftable pointer)
// CHECK-X64:    8 |     (B4 vbtable pointer)
// CHECK-X64:   16 |   struct B0 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B1 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)];
