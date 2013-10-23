// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct A4 {
	int a;
	A4() : a(0xf00000a4) {}
};

struct B4 {
	int a;
	B4() : a(0xf00000b4) {}
};

struct C4 {
	int a;
	C4() : a(0xf00000c4) {}
	virtual void f() {printf("C4");}
};

struct A16 {
	__declspec(align(16)) int a;
	A16() : a(0xf0000a16) {}
};

struct C16 {
	__declspec(align(16)) int a;
	C16() : a(0xf0000c16) {}
	virtual void f() {printf("C16");}
};

struct TestF0 : A4, virtual B4 {
	int a;
	TestF0() : a(0xf00000F0) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF0
// CHECK:    0 |   struct A4 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (TestF0 vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   12 |   struct B4 (virtual base)
// CHECK:   12 |     int a
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF0
// CHECK-X64:    0 |   struct A4 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (TestF0 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B4 (virtual base)
// CHECK-X64:   24 |     int a
// CHECK-X64:      | [sizeof=32, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct TestF1 : A4, virtual A16 {
	int a;
	TestF1() : a(0xf00000f1) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF1
// CHECK:    0 |   struct A4 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (TestF1 vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   16 |   struct A16 (virtual base)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=32, align=16
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF1
// CHECK-X64:    0 |   struct A4 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (TestF1 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   32 |   struct A16 (virtual base)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=48, align=16
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct TestF2 : A4, virtual C4 {
	int a;
	TestF2() : a(0xf00000f2) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF2
// CHECK:    0 |   struct A4 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (TestF2 vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   12 |   struct C4 (virtual base)
// CHECK:   12 |     (C4 vftable pointer)
// CHECK:   16 |     int a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF2
// CHECK-X64:    0 |   struct A4 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (TestF2 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct C4 (virtual base)
// CHECK-X64:   24 |     (C4 vftable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:      | [sizeof=40, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct TestF3 : A4, virtual C16 {
	int a;
	TestF3() : a(0xf00000f3) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF3
// CHECK:    0 |   struct A4 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (TestF3 vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   16 |   struct C16 (virtual base)
// CHECK:   16 |     (C16 vftable pointer)
// CHECK:   32 |     int a
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF3
// CHECK-X64:    0 |   struct A4 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (TestF3 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   32 |   struct C16 (virtual base)
// CHECK-X64:   32 |     (C16 vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct TestF4 : TestF3, A4 {
	int a;
	TestF4() : a(0xf00000f4) {}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF4
// CHECK:    0 |   struct TestF3 (base)
// CHECK:    0 |     struct A4 (base)
// CHECK:    0 |       int a
// CHECK:    4 |     (TestF3 vbtable pointer)
// CHECK:    8 |     int a
// CHECK:   12 |   struct A4 (base)
// CHECK:   12 |     int a
// CHECK:   16 |   int a
// CHECK:   32 |   struct C16 (virtual base)
// CHECK:   32 |     (C16 vftable pointer)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF4
// CHECK-X64:    0 |   struct TestF3 (base)
// CHECK-X64:    0 |     struct A4 (base)
// CHECK-X64:    0 |       int a
// CHECK-X64:    8 |     (TestF3 vbtable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct A4 (base)
// CHECK-X64:   24 |     int a
// CHECK-X64:   28 |   int a
// CHECK-X64:   32 |   struct C16 (virtual base)
// CHECK-X64:   32 |     (C16 vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct TestF5 : TestF3, A4 {
	int a;
	TestF5() : a(0xf00000f5) {}
	virtual void g() {printf("F5");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF5
// CHECK:    0 |   (TestF5 vftable pointer)
// CHECK:   16 |   struct TestF3 (base)
// CHECK:   16 |     struct A4 (base)
// CHECK:   16 |       int a
// CHECK:   20 |     (TestF3 vbtable pointer)
// CHECK:   24 |     int a
// CHECK:   28 |   struct A4 (base)
// CHECK:   28 |     int a
// CHECK:   32 |   int a
// CHECK:   48 |   struct C16 (virtual base)
// CHECK:   48 |     (C16 vftable pointer)
// CHECK:   64 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF5
// CHECK-X64:    0 |   (TestF5 vftable pointer)
// CHECK-X64:   16 |   struct TestF3 (base)
// CHECK-X64:   16 |     struct A4 (base)
// CHECK-X64:   16 |       int a
// CHECK-X64:   24 |     (TestF3 vbtable pointer)
// CHECK-X64:   32 |     int a
// CHECK-X64:   40 |   struct A4 (base)
// CHECK-X64:   40 |     int a
// CHECK-X64:   44 |   int a
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct TestF6 : TestF3, A4 {
	int a;
	TestF6() : a(0xf00000f6) {}
	virtual void f() {printf("F6");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF6
// CHECK:    0 |   struct TestF3 (base)
// CHECK:    0 |     struct A4 (base)
// CHECK:    0 |       int a
// CHECK:    4 |     (TestF3 vbtable pointer)
// CHECK:    8 |     int a
// CHECK:   12 |   struct A4 (base)
// CHECK:   12 |     int a
// CHECK:   16 |   int a
// CHECK:   44 |   (vtordisp for vbase C16)
// CHECK:   48 |   struct C16 (virtual base)
// CHECK:   48 |     (C16 vftable pointer)
// CHECK:   64 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF6
// CHECK-X64:    0 |   struct TestF3 (base)
// CHECK-X64:    0 |     struct A4 (base)
// CHECK-X64:    0 |       int a
// CHECK-X64:    8 |     (TestF3 vbtable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct A4 (base)
// CHECK-X64:   24 |     int a
// CHECK-X64:   28 |   int a
// CHECK-X64:   44 |   (vtordisp for vbase C16)
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct TestF7 : A4, virtual C16 {
	int a;
	TestF7() : a(0xf00000f7) {}
	virtual void f() {printf("F7");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF7
// CHECK:    0 |   struct A4 (base)
// CHECK:    0 |     int a
// CHECK:    4 |   (TestF7 vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   28 |   (vtordisp for vbase C16)
// CHECK:   32 |   struct C16 (virtual base)
// CHECK:   32 |     (C16 vftable pointer)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF7
// CHECK-X64:    0 |   struct A4 (base)
// CHECK-X64:    0 |     int a
// CHECK-X64:    8 |   (TestF7 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   44 |   (vtordisp for vbase C16)
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct TestF8 : TestF7, A4 {
	int a;
	TestF8() : a(0xf00000f8) {}
	virtual void f() {printf("F8");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF8
// CHECK:    0 |   struct TestF7 (base)
// CHECK:    0 |     struct A4 (base)
// CHECK:    0 |       int a
// CHECK:    4 |     (TestF7 vbtable pointer)
// CHECK:    8 |     int a
// CHECK:   12 |   struct A4 (base)
// CHECK:   12 |     int a
// CHECK:   16 |   int a
// CHECK:   44 |   (vtordisp for vbase C16)
// CHECK:   48 |   struct C16 (virtual base)
// CHECK:   48 |     (C16 vftable pointer)
// CHECK:   64 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF8
// CHECK-X64:    0 |   struct TestF7 (base)
// CHECK-X64:    0 |     struct A4 (base)
// CHECK-X64:    0 |       int a
// CHECK-X64:    8 |     (TestF7 vbtable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:   24 |   struct A4 (base)
// CHECK-X64:   24 |     int a
// CHECK-X64:   28 |   int a
// CHECK-X64:   44 |   (vtordisp for vbase C16)
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct TestF9 : A4, virtual C16 {
	int a;
	TestF9() : a(0xf00000f9) {}
	virtual void g() {printf("F9");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestF9
// CHECK:    0 |   (TestF9 vftable pointer)
// CHECK:    4 |   struct A4 (base)
// CHECK:    4 |     int a
// CHECK:    8 |   (TestF9 vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct C16 (virtual base)
// CHECK:   16 |     (C16 vftable pointer)
// CHECK:   32 |     int a
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestF9
// CHECK-X64:    0 |   (TestF9 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (TestF9 vbtable pointer)
// CHECK-X64:   24 |   int a
// CHECK-X64:   32 |   struct C16 (virtual base)
// CHECK-X64:   32 |     (C16 vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=32, nvalign=8]

struct TestFA : TestF9, A4 {
	int a;
	TestFA() : a(0xf00000fa) {}
	virtual void g() {printf("FA");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestFA
// CHECK:    0 |   struct TestF9 (primary base)
// CHECK:    0 |     (TestF9 vftable pointer)
// CHECK:    4 |     struct A4 (base)
// CHECK:    4 |       int a
// CHECK:    8 |     (TestF9 vbtable pointer)
// CHECK:   12 |     int a
// CHECK:   16 |   struct A4 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   int a
// CHECK:   32 |   struct C16 (virtual base)
// CHECK:   32 |     (C16 vftable pointer)
// CHECK:   48 |     int a
// CHECK:      | [sizeof=64, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestFA
// CHECK-X64:    0 |   struct TestF9 (primary base)
// CHECK-X64:    0 |     (TestF9 vftable pointer)
// CHECK-X64:    8 |     struct A4 (base)
// CHECK-X64:    8 |       int a
// CHECK-X64:   16 |     (TestF9 vbtable pointer)
// CHECK-X64:   24 |     int a
// CHECK-X64:   32 |   struct A4 (base)
// CHECK-X64:   32 |     int a
// CHECK-X64:   36 |   int a
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct TestFB : A16, virtual C16 {
	int a;
	TestFB() : a(0xf00000fb) {}
	virtual void g() {printf("Fb");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestFB
// CHECK:    0 |   (TestFB vftable pointer)
// CHECK:   16 |   struct A16 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   (TestFB vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct C16 (virtual base)
// CHECK:   64 |     (C16 vftable pointer)
// CHECK:   80 |     int a
// CHECK:      | [sizeof=96, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestFB
// CHECK-X64:    0 |   (TestFB vftable pointer)
// CHECK-X64:   16 |   struct A16 (base)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   (TestFB vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   48 |   struct C16 (virtual base)
// CHECK-X64:   48 |     (C16 vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct TestFC : TestFB, A4 {
	int a;
	TestFC() : a(0xf00000fc) {}
	virtual void g() {printf("FC");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct TestFC
// CHECK:    0 |   struct TestFB (primary base)
// CHECK:    0 |     (TestFB vftable pointer)
// CHECK:   16 |     struct A16 (base)
// CHECK:   16 |       int a
// CHECK:   32 |     (TestFB vbtable pointer)
// CHECK:   48 |     int a
// CHECK:   64 |   struct A4 (base)
// CHECK:   64 |     int a
// CHECK:   68 |   int a
// CHECK:   80 |   struct C16 (virtual base)
// CHECK:   80 |     (C16 vftable pointer)
// CHECK:   96 |     int a
// CHECK:      | [sizeof=112, align=16
// CHECK:      |  nvsize=80, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct TestFC
// CHECK-X64:    0 |   struct TestFB (primary base)
// CHECK-X64:    0 |     (TestFB vftable pointer)
// CHECK-X64:   16 |     struct A16 (base)
// CHECK-X64:   16 |       int a
// CHECK-X64:   32 |     (TestFB vbtable pointer)
// CHECK-X64:   40 |     int a
// CHECK-X64:   48 |   struct A4 (base)
// CHECK-X64:   48 |     int a
// CHECK-X64:   52 |   int a
// CHECK-X64:   64 |   struct C16 (virtual base)
// CHECK-X64:   64 |     (C16 vftable pointer)
// CHECK-X64:   80 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]


struct A16f {
	__declspec(align(16)) int a;
	A16f() : a(0xf0000a16) {}
	virtual void f() {printf("A16f");}
};

struct Y { char y; Y() : y(0xaa) {} };
struct X : virtual A16f {};

struct B : A4, Y, X {
	int a;
	B() : a(0xf000000b) {}
};

struct F0 : A4, B {
	int a;
	F0() : a(0xf00000f0) {}
	virtual void g() {printf("F0");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F0
// CHECK:    0 |   (F0 vftable pointer)
// CHECK:   16 |   struct A4 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   struct B (base)
// CHECK:   32 |     struct A4 (base)
// CHECK:   32 |       int a
// CHECK:   36 |     struct Y (base)
// CHECK:   36 |       char y
// CHECK:   48 |     struct X (base)
// CHECK:   48 |       (X vbtable pointer)
// CHECK:   52 |     int a
// CHECK:   64 |   int a
// CHECK:   80 |   struct A16f (virtual base)
// CHECK:   80 |     (A16f vftable pointer)
// CHECK:   96 |     int a
// CHECK:      | [sizeof=112, align=16
// CHECK:      |  nvsize=80, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F0
// CHECK-X64:    0 |   (F0 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   struct B (base)
// CHECK-X64:   16 |     struct A4 (base)
// CHECK-X64:   16 |       int a
// CHECK-X64:   20 |     struct Y (base)
// CHECK-X64:   20 |       char y
// CHECK-X64:   32 |     struct X (base)
// CHECK-X64:   32 |       (X vbtable pointer)
// CHECK-X64:   40 |     int a
// CHECK-X64:   48 |   int a
// CHECK-X64:   64 |   struct A16f (virtual base)
// CHECK-X64:   64 |     (A16f vftable pointer)
// CHECK-X64:   80 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]

struct F1 : B, A4 {
	int a;
	F1() : a(0xf00000f1) {}
	virtual void g() {printf("F1");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F1
// CHECK:    0 |   (F1 vftable pointer)
// CHECK:   16 |   struct B (base)
// CHECK:   16 |     struct A4 (base)
// CHECK:   16 |       int a
// CHECK:   20 |     struct Y (base)
// CHECK:   20 |       char y
// CHECK:   32 |     struct X (base)
// CHECK:   32 |       (X vbtable pointer)
// CHECK:   36 |     int a
// CHECK:   48 |   struct A4 (base)
// CHECK:   48 |     int a
// CHECK:   52 |   int a
// CHECK:   64 |   struct A16f (virtual base)
// CHECK:   64 |     (A16f vftable pointer)
// CHECK:   80 |     int a
// CHECK:      | [sizeof=96, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F1
// CHECK-X64:    0 |   (F1 vftable pointer)
// CHECK-X64:   16 |   struct B (base)
// CHECK-X64:   16 |     struct A4 (base)
// CHECK-X64:   16 |       int a
// CHECK-X64:   20 |     struct Y (base)
// CHECK-X64:   20 |       char y
// CHECK-X64:   32 |     struct X (base)
// CHECK-X64:   32 |       (X vbtable pointer)
// CHECK-X64:   40 |     int a
// CHECK-X64:   48 |   struct A4 (base)
// CHECK-X64:   48 |     int a
// CHECK-X64:   52 |   int a
// CHECK-X64:   64 |   struct A16f (virtual base)
// CHECK-X64:   64 |     (A16f vftable pointer)
// CHECK-X64:   80 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]

struct F2 : A4, virtual A16f {
	int a;
	F2() : a(0xf00000f2) {}
	virtual void g() {printf("F2");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F2
// CHECK:    0 |   (F2 vftable pointer)
// CHECK:    4 |   struct A4 (base)
// CHECK:    4 |     int a
// CHECK:    8 |   (F2 vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct A16f (virtual base)
// CHECK:   16 |     (A16f vftable pointer)
// CHECK:   32 |     int a
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F2
// CHECK-X64:    0 |   (F2 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (F2 vbtable pointer)
// CHECK-X64:   24 |   int a
// CHECK-X64:   32 |   struct A16f (virtual base)
// CHECK-X64:   32 |     (A16f vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=32, nvalign=8]

struct F3 : A4, virtual A16f {
	__declspec(align(16)) int a;
	F3() : a(0xf00000f3) {}
	virtual void g() {printf("F3");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F3
// CHECK:    0 |   (F3 vftable pointer)
// CHECK:   16 |   struct A4 (base)
// CHECK:   16 |     int a
// CHECK:   20 |   (F3 vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct A16f (virtual base)
// CHECK:   64 |     (A16f vftable pointer)
// CHECK:   80 |     int a
// CHECK:      | [sizeof=96, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F3
// CHECK-X64:    0 |   (F3 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (F3 vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct A16f (virtual base)
// CHECK-X64:   48 |     (A16f vftable pointer)
// CHECK-X64:   64 |     int a
// CHECK-X64:      | [sizeof=80, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct F4 : A4, B {
	__declspec(align(16)) int a;
	F4() : a(0xf00000f4) {}
	virtual void g() {printf("F4");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F4
// CHECK:    0 |   (F4 vftable pointer)
// CHECK:   16 |   struct A4 (base)
// CHECK:   16 |     int a
// CHECK:   32 |   struct B (base)
// CHECK:   32 |     struct A4 (base)
// CHECK:   32 |       int a
// CHECK:   36 |     struct Y (base)
// CHECK:   36 |       char y
// CHECK:   48 |     struct X (base)
// CHECK:   48 |       (X vbtable pointer)
// CHECK:   52 |     int a
// CHECK:   64 |   int a
// CHECK:   80 |   struct A16f (virtual base)
// CHECK:   80 |     (A16f vftable pointer)
// CHECK:   96 |     int a
// CHECK:      | [sizeof=112, align=16
// CHECK:      |  nvsize=80, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F4
// CHECK-X64:    0 |   (F4 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   struct B (base)
// CHECK-X64:   16 |     struct A4 (base)
// CHECK-X64:   16 |       int a
// CHECK-X64:   20 |     struct Y (base)
// CHECK-X64:   20 |       char y
// CHECK-X64:   32 |     struct X (base)
// CHECK-X64:   32 |       (X vbtable pointer)
// CHECK-X64:   40 |     int a
// CHECK-X64:   48 |   int a
// CHECK-X64:   64 |   struct A16f (virtual base)
// CHECK-X64:   64 |     (A16f vftable pointer)
// CHECK-X64:   80 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=64, nvalign=16]

struct F5 : A16f, virtual A4 {
	int a;
	F5() : a(0xf00000f5) {}
	virtual void g() {printf("F5");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F5
// CHECK:    0 |   struct A16f (primary base)
// CHECK:    0 |     (A16f vftable pointer)
// CHECK:   16 |     int a
// CHECK:   32 |   (F5 vbtable pointer)
// CHECK:   48 |   int a
// CHECK:   64 |   struct A4 (virtual base)
// CHECK:   64 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=64, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F5
// CHECK-X64:    0 |   struct A16f (primary base)
// CHECK-X64:    0 |     (A16f vftable pointer)
// CHECK-X64:   16 |     int a
// CHECK-X64:   32 |   (F5 vbtable pointer)
// CHECK-X64:   40 |   int a
// CHECK-X64:   48 |   struct A4 (virtual base)
// CHECK-X64:   48 |     int a
// CHECK-X64:      | [sizeof=64, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct F6 : virtual A16f, A4, virtual B {
	int a;
	F6() : a(0xf00000f6) {}
	virtual void g() {printf("F6");}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F6
// CHECK:    0 |   (F6 vftable pointer)
// CHECK:    4 |   struct A4 (base)
// CHECK:    4 |     int a
// CHECK:    8 |   (F6 vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct A16f (virtual base)
// CHECK:   16 |     (A16f vftable pointer)
// CHECK:   32 |     int a
// CHECK:   48 |   struct B (virtual base)
// CHECK:   48 |     struct A4 (base)
// CHECK:   48 |       int a
// CHECK:   52 |     struct Y (base)
// CHECK:   52 |       char y
// CHECK:   64 |     struct X (base)
// CHECK:   64 |       (X vbtable pointer)
// CHECK:   68 |     int a
// CHECK:      | [sizeof=80, align=16
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F6
// CHECK-X64:    0 |   (F6 vftable pointer)
// CHECK-X64:    8 |   struct A4 (base)
// CHECK-X64:    8 |     int a
// CHECK-X64:   16 |   (F6 vbtable pointer)
// CHECK-X64:   24 |   int a
// CHECK-X64:   32 |   struct A16f (virtual base)
// CHECK-X64:   32 |     (A16f vftable pointer)
// CHECK-X64:   48 |     int a
// CHECK-X64:   64 |   struct B (virtual base)
// CHECK-X64:   64 |     struct A4 (base)
// CHECK-X64:   64 |       int a
// CHECK-X64:   68 |     struct Y (base)
// CHECK-X64:   68 |       char y
// CHECK-X64:   80 |     struct X (base)
// CHECK-X64:   80 |       (X vbtable pointer)
// CHECK-X64:   88 |     int a
// CHECK-X64:      | [sizeof=96, align=16
// CHECK-X64:      |  nvsize=32, nvalign=8]

int a[
sizeof(TestF0)+
sizeof(TestF1)+
sizeof(TestF2)+
sizeof(TestF3)+
sizeof(TestF4)+
sizeof(TestF5)+
sizeof(TestF6)+
sizeof(TestF7)+
sizeof(TestF8)+
sizeof(TestF9)+
sizeof(TestFA)+
sizeof(TestFB)+
sizeof(TestFC)+
sizeof(F0)+
sizeof(F1)+
sizeof(F2)+
sizeof(F3)+
sizeof(F4)+
sizeof(F5)+
sizeof(F6)];
