// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);
char buffer[419430400];

struct A {
	char a;
	A() {
		printf("A   = %d\n", (int)((char*)this - buffer));
		printf("A.a = %d\n", (int)((char*)&a - buffer));
	}
};

struct B {
	__declspec(align(4)) long long a;
	B() {
		printf("B   = %d\n", (int)((char*)this - buffer));
		printf("B.a = %d\n", (int)((char*)&a - buffer));
	}
};

#pragma pack(push, 2)
struct X {
	B a;
	char b;
	int c;
	X() {
		printf("X   = %d\n", (int)((char*)this - buffer));
		printf("X.a = %d\n", (int)((char*)&a - buffer));
		printf("X.b = %d\n", (int)((char*)&b - buffer));
		printf("X.c = %d\n", (int)((char*)&c - buffer));
	}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct X
// CHECK:    0 |   struct B a
// CHECK:    0 |     long long a
// CHECK:    8 |   char b
// CHECK:   10 |   int c
// CHECK:      | [sizeof=16, align=4
// CHECK:      |  nvsize=16, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct X
// CHECK-X64:    0 |   struct B a
// CHECK-X64:    0 |     long long a
// CHECK-X64:    8 |   char b
// CHECK-X64:   10 |   int c
// CHECK-X64:      | [sizeof=16, align=4
// CHECK-X64:      |  nvsize=16, nvalign=4]

struct Y : A, B {
	char a;
	int b;
	Y() {
		printf("Y   = %d\n", (int)((char*)this - buffer));
		printf("Y.a = %d\n", (int)((char*)&a - buffer));
		printf("Y.b = %d\n", (int)((char*)&b - buffer));
	}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct Y
// CHECK:    0 |   struct A (base)
// CHECK:    0 |     char a
// CHECK:    4 |   struct B (base)
// CHECK:    4 |     long long a
// CHECK:   12 |   char a
// CHECK:   14 |   int b
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=20, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct Y
// CHECK-X64:    0 |   struct A (base)
// CHECK-X64:    0 |     char a
// CHECK-X64:    4 |   struct B (base)
// CHECK-X64:    4 |     long long a
// CHECK-X64:   12 |   char a
// CHECK-X64:   14 |   int b
// CHECK-X64:      | [sizeof=20, align=4
// CHECK-X64:      |  nvsize=20, nvalign=4]

struct Z : virtual B {
	char a;
	int b;
	Z() {
		printf("Z   = %d\n", (int)((char*)this - buffer));
		printf("Z.a = %d\n", (int)((char*)&a - buffer));
		printf("Z.b = %d\n", (int)((char*)&b - buffer));
	}
};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct Z
// CHECK:    0 |   (Z vbtable pointer)
// CHECK:    4 |   char a
// CHECK:    6 |   int b
// CHECK:   12 |   struct B (virtual base)
// CHECK:   12 |     long long a
// CHECK:      | [sizeof=20, align=4
// CHECK:      |  nvsize=10, nvalign=2]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct Z
// CHECK-X64:    0 |   (Z vbtable pointer)
// CHECK-X64:    8 |   char a
// CHECK-X64:   10 |   int b
// CHECK-X64:   16 |   struct B (virtual base)
// CHECK-X64:   16 |     long long a
// CHECK-X64:      | [sizeof=24, align=4
// CHECK-X64:      |  nvsize=14, nvalign=2]

#pragma pack(pop)

struct A1 { long long a; };
#pragma pack(push, 1)
struct B1 : virtual A1 { char a; };
#pragma pack(pop)
struct C1 : B1 {};

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C1
// CHECK:    0 |   struct B1 (base)
// CHECK:    0 |     (B1 vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    8 |   struct A1 (virtual base)
// CHECK:    8 |     long long a
// CHECK:      | [sizeof=16, align=8
// CHECK:      |  nvsize=5, nvalign=1]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C1
// CHECK-X64:    0 |   struct B1 (base)
// CHECK-X64:    0 |     (B1 vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   16 |   struct A1 (virtual base)
// CHECK-X64:   16 |     long long a
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=9, nvalign=1]

struct CA0 {
	CA0() {}
};
struct CA1 : virtual CA0 {
	CA1() {}
};
#pragma pack(push, 1)
struct CA2 : public CA1, public CA0 {
	virtual void CA2Method() {}
	CA2() {}
};
#pragma pack(pop)

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct CA2
// CHECK:    0 |   (CA2 vftable pointer)
// CHECK:    4 |   struct CA1 (base)
// CHECK:    4 |     (CA1 vbtable pointer)
// CHECK:    9 |   struct CA0 (base) (empty)
// CHECK:    9 |   struct CA0 (virtual base) (empty)
// CHECK:      | [sizeof=9, align=1
// CHECK:      |  nvsize=9, nvalign=1]
// CHECK-C64: *** Dumping AST Record Layout
// CHECK-C64:    0 | struct CA2
// CHECK-C64:    0 |   (CA2 vftable pointer)
// CHECK-C64:    8 |   struct CA1 (base)
// CHECK-C64:    8 |     (CA1 vbtable pointer)
// CHECK-C64:   17 |   struct CA0 (base) (empty)
// CHECK-C64:   17 |   struct CA0 (virtual base) (empty)
// CHECK-C64:      | [sizeof=17, align=1
// CHECK-C64:      |  nvsize=17, nvalign=1]

int a[
sizeof(X)+
sizeof(Y)+
sizeof(Z)+
sizeof(C1)+
sizeof(CA2)];
