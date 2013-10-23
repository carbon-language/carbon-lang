// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { B0() { printf("B0 = %p\n", this); } };
struct B1 { B1() { printf("B1 = %p\n", this); } };
struct B2 { B2() { printf("B2 = %p\n", this); } };
struct B3 { B3() { printf("B3 = %p\n", this); } };
struct B4 { B4() { printf("B4 = %p\n", this); } };
struct B5 { B5() { printf("B5 = %p\n", this); } };
struct __declspec(align(2)) B6 { B6() { printf("B6 = %p\n", this); } };
struct __declspec(align(16)) B7 { B7() { printf("B7 = %p\n", this); } };
struct B8 { char c[5]; B8() { printf("B8 = %p\n", this); } };
struct B9 { char c[6]; B9() { printf("B9 = %p\n", this); } };
struct B10 { char c[7]; B10() { printf("B10 = %p\n", this); } };
struct B11 { char c[8]; B11() { printf("B11 = %p\n", this); } };
struct B0X { B0X() { printf("B0 = %p\n", this); } };
struct B1X { B1X() { printf("B1 = %p\n", this); } };
struct __declspec(align(16)) B2X { B2X() { printf("B2 = %p\n", this); } };
struct __declspec(align(2)) B3X { B3X() { printf("B3 = %p\n", this); } };
struct B4X { B4X() { printf("B4 = %p\n", this); } };
struct B5X { B5X() { printf("B5 = %p\n", this); } };
struct B6X { B6X() { printf("B6 = %p\n", this); } };
struct B8X { short a; B8X() : a(0xf00000B8) { printf("B8 = %p\n", this); } };

struct AA : B8, B1, virtual B0 {
	int a;
	AA() : a(0xf00000AA) { printf("AA = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AA
// CHECK:    0 |   struct B8 (base)
// CHECK:    0 |     char [5] c
// CHECK:   13 |   struct B1 (base) (empty)
// CHECK:    8 |   (AA vbtable pointer)
// CHECK:   16 |   int a
// CHECK:   20 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AA
// CHECK-X64:    0 |   struct B8 (base)
// CHECK-X64:    0 |     char [5] c
// CHECK-X64:   17 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AA vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AB : B8, B1, virtual B0 {
	short a;
	AB() : a(0xf00000AB) { printf("AB = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AB
// CHECK:    0 |   struct B8 (base)
// CHECK:    0 |     char [5] c
// CHECK:   13 |   struct B1 (base) (empty)
// CHECK:    8 |   (AB vbtable pointer)
// CHECK:   14 |   short a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AB
// CHECK-X64:    0 |   struct B8 (base)
// CHECK-X64:    0 |     char [5] c
// CHECK-X64:   17 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AB vbtable pointer)
// CHECK-X64:   18 |   short a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AC : B8, B1, virtual B0 {
	char a;
	AC() : a(0xf00000AC) { printf("AC = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AC
// CHECK:    0 |   struct B8 (base)
// CHECK:    0 |     char [5] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AC vbtable pointer)
// CHECK:   12 |   char a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AC
// CHECK-X64:    0 |   struct B8 (base)
// CHECK-X64:    0 |     char [5] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AC vbtable pointer)
// CHECK-X64:   16 |   char a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AD : B8, B1, virtual B0 {
	AD() { printf("AD = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AD
// CHECK:    0 |   struct B8 (base)
// CHECK:    0 |     char [5] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AD vbtable pointer)
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AD
// CHECK-X64:    0 |   struct B8 (base)
// CHECK-X64:    0 |     char [5] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AD vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct AA1 : B9, B1, virtual B0 {
	int a;
	AA1() : a(0xf0000AA1) { printf("AA1 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AA1
// CHECK:    0 |   struct B9 (base)
// CHECK:    0 |     char [6] c
// CHECK:   14 |   struct B1 (base) (empty)
// CHECK:    8 |   (AA1 vbtable pointer)
// CHECK:   16 |   int a
// CHECK:   20 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AA1
// CHECK-X64:    0 |   struct B9 (base)
// CHECK-X64:    0 |     char [6] c
// CHECK-X64:   18 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AA1 vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AB1 : B9, B1, virtual B0 {
	short a;
	AB1() : a(0xf0000AB1) { printf("AB1 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AB1
// CHECK:    0 |   struct B9 (base)
// CHECK:    0 |     char [6] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AB1 vbtable pointer)
// CHECK:   12 |   short a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AB1
// CHECK-X64:    0 |   struct B9 (base)
// CHECK-X64:    0 |     char [6] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AB1 vbtable pointer)
// CHECK-X64:   16 |   short a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AC1 : B9, B1, virtual B0 {
	char a;
	AC1() : a(0xf0000AC1) { printf("AC1 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AC1
// CHECK:    0 |   struct B9 (base)
// CHECK:    0 |     char [6] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AC1 vbtable pointer)
// CHECK:   12 |   char a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AC1
// CHECK-X64:    0 |   struct B9 (base)
// CHECK-X64:    0 |     char [6] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AC1 vbtable pointer)
// CHECK-X64:   16 |   char a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AD1 : B9, B1, virtual B0 {
	AD1() { printf("AD1 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AD1
// CHECK:    0 |   struct B9 (base)
// CHECK:    0 |     char [6] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AD1 vbtable pointer)
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AD1
// CHECK-X64:    0 |   struct B9 (base)
// CHECK-X64:    0 |     char [6] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AD1 vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct AA2 : B10, B1, virtual B0 {
	int a;
	AA2() : a(0xf0000AA2) { printf("AA2 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AA2
// CHECK:    0 |   struct B10 (base)
// CHECK:    0 |     char [7] c
// CHECK:   15 |   struct B1 (base) (empty)
// CHECK:    8 |   (AA2 vbtable pointer)
// CHECK:   16 |   int a
// CHECK:   20 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AA2
// CHECK-X64:    0 |   struct B10 (base)
// CHECK-X64:    0 |     char [7] c
// CHECK-X64:   19 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AA2 vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AB2 : B10, B1, virtual B0 {
	short a;
	AB2() : a(0xf0000AB2) { printf("AB2 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AB2
// CHECK:    0 |   struct B10 (base)
// CHECK:    0 |     char [7] c
// CHECK:   13 |   struct B1 (base) (empty)
// CHECK:    8 |   (AB2 vbtable pointer)
// CHECK:   14 |   short a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AB2
// CHECK-X64:    0 |   struct B10 (base)
// CHECK-X64:    0 |     char [7] c
// CHECK-X64:   17 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AB2 vbtable pointer)
// CHECK-X64:   18 |   short a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AC2 : B10, B1, virtual B0 {
	char a;
	AC2() : a(0xf0000AC2) { printf("AC2 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AC2
// CHECK:    0 |   struct B10 (base)
// CHECK:    0 |     char [7] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AC2 vbtable pointer)
// CHECK:   12 |   char a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AC2
// CHECK-X64:    0 |   struct B10 (base)
// CHECK-X64:    0 |     char [7] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AC2 vbtable pointer)
// CHECK-X64:   16 |   char a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AD2 : B10, B1, virtual B0 {
	AD2() { printf("AD2 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AD2
// CHECK:    0 |   struct B10 (base)
// CHECK:    0 |     char [7] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AD2 vbtable pointer)
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AD2
// CHECK-X64:    0 |   struct B10 (base)
// CHECK-X64:    0 |     char [7] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AD2 vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct AA3 : B11, B1, virtual B0 {
	int a;
	AA3() : a(0xf0000AA3) { printf("AA3 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AA3
// CHECK:    0 |   struct B11 (base)
// CHECK:    0 |     char [8] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AA3 vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AA3
// CHECK-X64:    0 |   struct B11 (base)
// CHECK-X64:    0 |     char [8] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AA3 vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AB3 : B11, B1, virtual B0 {
	short a;
	AB3() : a(0xf0000AB3) { printf("AB3 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AB3
// CHECK:    0 |   struct B11 (base)
// CHECK:    0 |     char [8] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AB3 vbtable pointer)
// CHECK:   12 |   short a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AB3
// CHECK-X64:    0 |   struct B11 (base)
// CHECK-X64:    0 |     char [8] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AB3 vbtable pointer)
// CHECK-X64:   16 |   short a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AC3 : B11, B1, virtual B0 {
	char a;
	AC3() : a(0xf0000AC3) { printf("AC3 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AC3
// CHECK:    0 |   struct B11 (base)
// CHECK:    0 |     char [8] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AC3 vbtable pointer)
// CHECK:   12 |   char a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AC3
// CHECK-X64:    0 |   struct B11 (base)
// CHECK-X64:    0 |     char [8] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AC3 vbtable pointer)
// CHECK-X64:   16 |   char a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct AD3 : B11, B1, virtual B0 {
	AD3() { printf("AD3 = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AD3
// CHECK:    0 |   struct B11 (base)
// CHECK:    0 |     char [8] c
// CHECK:   12 |   struct B1 (base) (empty)
// CHECK:    8 |   (AD3 vbtable pointer)
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AD3
// CHECK-X64:    0 |   struct B11 (base)
// CHECK-X64:    0 |     char [8] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (AD3 vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct B : B1, B2, virtual B0 {
	B() { printf("B = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   struct B1 (base) (empty)
// CHECK:    8 |   struct B2 (base) (empty)
// CHECK:    4 |   (B vbtable pointer)
// CHECK:    8 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=8, align=4
// CHECK:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   struct B1 (base) (empty)
// CHECK-X64:   16 |   struct B2 (base) (empty)
// CHECK-X64:    8 |   (B vbtable pointer)
// CHECK-X64:   16 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=16, align=8
// CHECK-X64:      |  nvsize=16, nvalign=8]

struct C : B1, B2, B3, virtual B0 {
	char a;
	C() : a(0xf000000C) { printf("C = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   struct B1 (base) (empty)
// CHECK:    1 |   struct B2 (base) (empty)
// CHECK:    8 |   struct B3 (base) (empty)
// CHECK:    4 |   (C vbtable pointer)
// CHECK:    8 |   char a
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   struct B1 (base) (empty)
// CHECK-X64:    1 |   struct B2 (base) (empty)
// CHECK-X64:   16 |   struct B3 (base) (empty)
// CHECK-X64:    8 |   (C vbtable pointer)
// CHECK-X64:   16 |   char a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct D : B1, B2, B3, B4, B5, virtual B0 {
	int a;
	D() : a(0xf000000D) { printf("D = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct B1 (base) (empty)
// CHECK:    1 |   struct B2 (base) (empty)
// CHECK:    2 |   struct B3 (base) (empty)
// CHECK:    3 |   struct B4 (base) (empty)
// CHECK:    8 |   struct B5 (base) (empty)
// CHECK:    4 |   (D vbtable pointer)
// CHECK:    8 |   int a
// CHECK:   12 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   struct B1 (base) (empty)
// CHECK-X64:    1 |   struct B2 (base) (empty)
// CHECK-X64:    2 |   struct B3 (base) (empty)
// CHECK-X64:    3 |   struct B4 (base) (empty)
// CHECK-X64:   16 |   struct B5 (base) (empty)
// CHECK-X64:    8 |   (D vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct E : B1, B6, B3, B4, B5, virtual B0 {
	int a;
	E() : a(0xf000000E) { printf("E = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   struct B1 (base) (empty)
// CHECK:    2 |   struct B6 (base) (empty)
// CHECK:    3 |   struct B3 (base) (empty)
// CHECK:    4 |   struct B4 (base) (empty)
// CHECK:   13 |   struct B5 (base) (empty)
// CHECK:    8 |   (E vbtable pointer)
// CHECK:   16 |   int a
// CHECK:   20 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   struct B1 (base) (empty)
// CHECK-X64:    2 |   struct B6 (base) (empty)
// CHECK-X64:    3 |   struct B3 (base) (empty)
// CHECK-X64:    4 |   struct B4 (base) (empty)
// CHECK-X64:   17 |   struct B5 (base) (empty)
// CHECK-X64:    8 |   (E vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct F : B1, B6, B4, B8, B5, virtual B0 {
	int a;
	F() : a(0xf000000F) { printf("&a = %p\n", &a); printf("F = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   struct B1 (base) (empty)
// CHECK:    2 |   struct B6 (base) (empty)
// CHECK:    3 |   struct B4 (base) (empty)
// CHECK:    3 |   struct B8 (base)
// CHECK:    3 |     char [5] c
// CHECK:   12 |   struct B5 (base) (empty)
// CHECK:    8 |   (F vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   struct B1 (base) (empty)
// CHECK-X64:    2 |   struct B6 (base) (empty)
// CHECK-X64:    3 |   struct B4 (base) (empty)
// CHECK-X64:    3 |   struct B8 (base)
// CHECK-X64:    3 |     char [5] c
// CHECK-X64:   16 |   struct B5 (base) (empty)
// CHECK-X64:    8 |   (F vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   24 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct G : B8, B1, virtual B0 {
	int a;
	__declspec(align(16)) int a1;
	G() : a(0xf0000010), a1(0xf0000010) { printf("G = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct G
// CHECK:    0 |   struct B8 (base)
// CHECK:    0 |     char [5] c
// CHECK:   21 |   struct B1 (base) (empty)
// CHECK:    8 |   (G vbtable pointer)
// CHECK:   24 |   int a
// CHECK:   32 |   int a1
// CHECK:   48 |   struct B0 (virtual base) (empty)
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct G
// CHECK-X64:    0 |   struct B8 (base)
// CHECK-X64:    0 |     char [5] c
// CHECK-X64:   16 |   struct B1 (base) (empty)
// CHECK-X64:    8 |   (G vbtable pointer)
// CHECK-X64:   16 |   int a
// CHECK-X64:   32 |   int a1
// CHECK-X64:   48 |   struct B0 (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct AX : B1X, B2X, B3X, B4X, virtual B0X {
	int a;
	AX() : a(0xf000000A) { printf(" A = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct AX
// CHECK:    0 |   struct B1X (base) (empty)
// CHECK:   16 |   struct B2X (base) (empty)
// CHECK:   18 |   struct B3X (base) (empty)
// CHECK:   35 |   struct B4X (base) (empty)
// CHECK:   20 |   (AX vbtable pointer)
// CHECK:   36 |   int a
// CHECK:   48 |   struct B0X (virtual base) (empty)
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct AX
// CHECK-X64:    0 |   struct B1X (base) (empty)
// CHECK-X64:   16 |   struct B2X (base) (empty)
// CHECK-X64:   18 |   struct B3X (base) (empty)
// CHECK-X64:   33 |   struct B4X (base) (empty)
// CHECK-X64:   24 |   (AX vbtable pointer)
// CHECK-X64:   36 |   int a
// CHECK-X64:   48 |   struct B0X (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct BX : B2X, B1X, B3X, B4X, virtual B0X {
	int a;
	BX() : a(0xf000000B) { printf(" B = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct BX
// CHECK:    0 |   struct B2X (base) (empty)
// CHECK:    1 |   struct B1X (base) (empty)
// CHECK:    2 |   struct B3X (base) (empty)
// CHECK:   19 |   struct B4X (base) (empty)
// CHECK:    4 |   (BX vbtable pointer)
// CHECK:   20 |   int a
// CHECK:   32 |   struct B0X (virtual base) (empty)
// CHECK:      | [sizeof=32, align=16
// CHECK:      |  nvsize=32, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct BX
// CHECK-X64:    0 |   struct B2X (base) (empty)
// CHECK-X64:    1 |   struct B1X (base) (empty)
// CHECK-X64:    2 |   struct B3X (base) (empty)
// CHECK-X64:   17 |   struct B4X (base) (empty)
// CHECK-X64:    8 |   (BX vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   32 |   struct B0X (virtual base) (empty)
// CHECK-X64:      | [sizeof=32, align=16
// CHECK-X64:      |  nvsize=32, nvalign=16]

struct CX : B1X, B3X, B2X, virtual B0X {
	int a;
	CX() : a(0xf000000C) { printf(" C = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct CX
// CHECK:    0 |   struct B1X (base) (empty)
// CHECK:    2 |   struct B3X (base) (empty)
// CHECK:   32 |   struct B2X (base) (empty)
// CHECK:    4 |   (CX vbtable pointer)
// CHECK:   32 |   int a
// CHECK:   48 |   struct B0X (virtual base) (empty)
// CHECK:      | [sizeof=48, align=16
// CHECK:      |  nvsize=48, nvalign=16]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct CX
// CHECK-X64:    0 |   struct B1X (base) (empty)
// CHECK-X64:    2 |   struct B3X (base) (empty)
// CHECK-X64:   32 |   struct B2X (base) (empty)
// CHECK-X64:    8 |   (CX vbtable pointer)
// CHECK-X64:   32 |   int a
// CHECK-X64:   48 |   struct B0X (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=16
// CHECK-X64:      |  nvsize=48, nvalign=16]

struct DX : B8X, B1X, virtual B0X {
	int a;
	DX() : a(0xf000000D) { printf(" D = %p\n", this); }
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct DX
// CHECK:    0 |   struct B8X (base)
// CHECK:    0 |     short a
// CHECK:   10 |   struct B1X (base) (empty)
// CHECK:    4 |   (DX vbtable pointer)
// CHECK:   12 |   int a
// CHECK:   16 |   struct B0X (virtual base) (empty)
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct DX
// CHECK-X64:    0 |   struct B8X (base)
// CHECK-X64:    0 |     short a
// CHECK-X64:   18 |   struct B1X (base) (empty)
// CHECK-X64:    8 |   (DX vbtable pointer)
// CHECK-X64:   20 |   int a
// CHECK-X64:   24 |   struct B0X (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

int a[
sizeof(AA)+
sizeof(AB)+
sizeof(AC)+
sizeof(AD)+
sizeof(AA1)+
sizeof(AB1)+
sizeof(AC1)+
sizeof(AD1)+
sizeof(AA2)+
sizeof(AB2)+
sizeof(AC2)+
sizeof(AD2)+
sizeof(AA3)+
sizeof(AB3)+
sizeof(AC3)+
sizeof(AD3)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(G)+
sizeof(AX)+
sizeof(BX)+
sizeof(CX)+
sizeof(DX)];
