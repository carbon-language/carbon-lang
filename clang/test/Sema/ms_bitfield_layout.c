// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

typedef struct A {
	char x;
	int a : 22;
	int : 0;
	int c : 10;
	char b : 3;
	char d: 4;
	short y;
} A;

// CHECK: Type: struct A
// CHECK:   Size:128
// CHECK:   Alignment:32
// CHECK:   FieldOffsets: [0, 32, 64, 64, 96, 99, 112]>

typedef struct B {
	char x;
	int : 0;
	short a : 4;
	char y;
} B;

// CHECK: Type: struct B
// CHECK:   Size:48
// CHECK:   Alignment:16
// CHECK:   FieldOffsets: [0, 8, 16, 32]>

typedef struct C {
	char x;
	short a : 4;
	int : 0;
	char y;
} C;

// CHECK: Type: struct C
// CHECK:   Size:64
// CHECK:   Alignment:32
// CHECK:   FieldOffsets: [0, 16, 32, 32]>

typedef struct D {
	char x;
	short : 0;
	int : 0;
	char y;
} D;

// CHECK: Type: struct D
// CHECK:   Size:16
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 8, 8]>

typedef union E {
	char x;
	long long a : 3;
	int b : 3;
	long long : 0;
	short y;
} E;

// CHECK: Type: union E
// CHECK:   Size:64
// CHECK:   Alignment:16
// CHECK:   FieldOffsets: [0, 0, 0, 0, 0]>

typedef struct F {
	char x;
	char a : 3;
	char b : 3;
	char c : 3;
	short d : 6;
	short e : 6;
	short f : 6;
	short g : 11;
	short h : 11;
	short i : 11;
	short y;
} F;

// CHECK: Type: struct F
// CHECK:   Size:128
// CHECK:   Alignment:16
// CHECK:   FieldOffsets: [0, 8, 11, 16, 32, 38, 48, 64, 80, 96, 112]>

typedef union G {
	char x;
	int a : 3;
	int : 0;
	long long : 0;
	short y;
} G;

// CHECK: Type: union G
// CHECK:   Size:32
// CHECK:   Alignment:16
// CHECK:   FieldOffsets: [0, 0, 0, 0, 0]>

typedef struct H {
	unsigned short a : 1;
	unsigned char : 0;
	unsigned long : 0;
	unsigned short c : 1;
} H;

// CHECK: Type: struct H
// CHECK:   Size:32
// CHECK:   Alignment:16
// CHECK:   FieldOffsets: [0, 16, 16, 16]>

#pragma pack(push, 1)

typedef struct A1 {
	char x;
	int a : 22;
	int : 0;
	int c : 10;
	char b : 3;
	char d: 4;
	short y;
} A1;

// CHECK: Type: struct A1
// CHECK:   Size:96
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 40, 40, 72, 75, 80]>

typedef struct B1 {
	char x;
	int : 0;
	short a : 4;
	char y;
} B1;

// CHECK: Type: struct B1
// CHECK:   Size:32
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 8, 24]>

typedef struct C1 {
	char x;
	short a : 4;
	int : 0;
	char y;
} C1;

// CHECK: Type: struct C1
// CHECK:   Size:32
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 24, 24]>

typedef struct D1 {
	char x;
	short : 0;
	int : 0;
	char y;
} D1;

// CHECK: Type: struct D1
// CHECK:   Size:16
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 8, 8]>

typedef union E1 {
	char x;
	long long a : 3;
	int b : 3;
	long long : 0;
	short y;
} E1;

// CHECK: Type: union E1
// CHECK:   Size:64
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 0, 0, 0, 0]>

typedef struct F1 {
	char x;
	char a : 3;
	char b : 3;
	char c : 3;
	short d : 6;
	short e : 6;
	short f : 6;
	short g : 11;
	short h : 11;
	short i : 11;
	short y;
} F1;

// CHECK: Type: struct F1
// CHECK:   Size:120
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 8, 11, 16, 24, 30, 40, 56, 72, 88, 104]>

typedef union G1 {
	char x;
	int a : 3;
	int : 0;
	long long : 0;
	short y;
} G1;

// CHECK: Type: union G1
// CHECK:   Size:32
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 0, 0, 0, 0]>

typedef struct H1 {
	unsigned long a : 1;
	unsigned char : 0;
	unsigned long : 0;
	unsigned long c : 1;
} H1;

// CHECK: Type: struct H1
// CHECK:   Size:64
// CHECK:   Alignment:8
// CHECK:   FieldOffsets: [0, 32, 32, 32]>

#pragma pack(pop)

int x[
sizeof(A ) +
sizeof(B ) +
sizeof(C ) +
sizeof(D ) +
sizeof(E ) +
sizeof(F ) +
sizeof(G ) +
sizeof(H ) +
sizeof(A1) +
sizeof(B1) +
sizeof(C1) +
sizeof(D1) +
sizeof(E1) +
sizeof(F1) +
sizeof(G1) +
sizeof(H1) +
0];
