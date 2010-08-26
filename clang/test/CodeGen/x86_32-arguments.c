// RUN: %clang_cc1 -w -fblocks -triple i386-apple-darwin9 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK: define signext i8 @f0()
char f0(void) {
  return 0;
}

// CHECK: define signext i16 @f1()
short f1(void) {
  return 0;
}

// CHECK: define i32 @f2()
int f2(void) {
  return 0;
}

// CHECK: define float @f3()
float f3(void) {
  return 0;
}

// CHECK: define double @f4()
double f4(void) {
  return 0;
}

// CHECK: define x86_fp80 @f5()
long double f5(void) {
  return 0;
}

// CHECK: define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8* %a4)
void f6(char a0, short a1, int a2, long long a3, void *a4) {}

// CHECK: define void @f7(i32 %a0)
typedef enum { A, B, C } e7;
void f7(e7 a0) {}

// CHECK: define i64 @f8_1()
// CHECK: define void @f8_2(i32 %a0.0, i32 %a0.1)
struct s8 {
  int a;
  int b;
};
struct s8 f8_1(void) { while (1) {} }
void f8_2(struct s8 a0) {}

// This should be passed just as s8.

// CHECK: define i64 @f9_1()

// FIXME: llvm-gcc expands this, this may have some value for the
// backend in terms of optimization but doesn't change the ABI.
// CHECK: define void @f9_2(%struct.s9* byval %a0)
struct s9 {
  int a : 17;
  int b;
};
struct s9 f9_1(void) { while (1) {} }
void f9_2(struct s9 a0) {}

// Return of small structures and unions

// CHECK: float @f10()
struct s10 {
  union { };
  float f;
} f10(void) { while (1) {} }

// Small vectors and 1 x {i64,double} are returned in registers

// CHECK: i32 @f11()
// CHECK: void @f12(<2 x i32>* sret %agg.result)
// CHECK: i64 @f13()
// CHECK: i64 @f14()
// CHECK: <2 x i64> @f15()
// CHECK: <2 x i64> @f16()
typedef short T11 __attribute__ ((vector_size (4)));
T11 f11(void) { while (1) {} }
typedef int T12 __attribute__ ((vector_size (8)));
T12 f12(void) { while (1) {} }
typedef long long T13 __attribute__ ((vector_size (8)));
T13 f13(void) { while (1) {} }
typedef double T14 __attribute__ ((vector_size (8)));
T14 f14(void) { while (1) {} }
typedef long long T15 __attribute__ ((vector_size (16)));
T15 f15(void) { while (1) {} }
typedef double T16 __attribute__ ((vector_size (16)));
T16 f16(void) { while (1) {} }

// And when the single element in a struct (but not for 64 and
// 128-bits).

// CHECK: i32 @f17()
// CHECK: void @f18(%2* sret %agg.result)
// CHECK: void @f19(%3* sret %agg.result)
// CHECK: void @f20(%4* sret %agg.result)
// CHECK: void @f21(%5* sret %agg.result)
// CHECK: void @f22(%6* sret %agg.result)
struct { T11 a; } f17(void) { while (1) {} }
struct { T12 a; } f18(void) { while (1) {} }
struct { T13 a; } f19(void) { while (1) {} }
struct { T14 a; } f20(void) { while (1) {} }
struct { T15 a; } f21(void) { while (1) {} }
struct { T16 a; } f22(void) { while (1) {} }

// Single element structures are handled specially

// CHECK: float @f23()
// CHECK: float @f24()
// CHECK: float @f25()
struct { float a; } f23(void) { while (1) {} }
struct { float a[1]; } f24(void) { while (1) {} }
struct { struct {} a; struct { float a[1]; } b; } f25(void) { while (1) {} }

// Small structures are handled recursively
// CHECK: i32 @f26()
// CHECK: void @f27(%struct.s27* sret %agg.result)
struct s26 { struct { char a, b; } a; struct { char a, b; } b; } f26(void) { while (1) {} }
struct s27 { struct { char a, b, c; } a; struct { char a; } b; } f27(void) { while (1) {} }

// CHECK: void @f28(%struct.s28* sret %agg.result)
struct s28 { int a; int b[]; } f28(void) { while (1) {} }

// CHECK: define i16 @f29()
struct s29 { struct { } a[1]; char b; char c; } f29(void) { while (1) {} }

// CHECK: define i16 @f30()
struct s30 { char a; char b : 4; } f30(void) { while (1) {} }

// CHECK: define float @f31()
struct s31 { char : 0; float b; char : 0; } f31(void) { while (1) {} }

// CHECK: define i32 @f32()
struct s32 { char a; unsigned : 0; } f32(void) { while (1) {} }

// CHECK: define float @f33()
struct s33 { float a; long long : 0; } f33(void) { while (1) {} }

// CHECK: define float @f34()
struct s34 { struct { int : 0; } a; float b; } f34(void) { while (1) {} }

// CHECK: define i16 @f35()
struct s35 { struct { int : 0; } a; char b; char c; } f35(void) { while (1) {} }

// CHECK: define i16 @f36()
struct s36 { struct { int : 0; } a[2][10]; char b; char c; } f36(void) { while (1) {} }

// CHECK: define float @f37()
struct s37 { float c[1][1]; } f37(void) { while (1) {} }

// CHECK: define void @f38(%struct.s38* sret %agg.result)
struct s38 { char a[3]; short b; } f38(void) { while (1) {} }

// CHECK: define void @f39(%struct.s39* byval align 16 %x)
typedef int v39 __attribute((vector_size(16)));
struct s39 { v39 x; };
void f39(struct s39 x) {}

// <rdar://problem/7247671>
// CHECK: define i32 @f40()
enum e40 { ec0 = 0 };
enum e40 f40(void) { }

// CHECK: define void ()* @f41()
typedef void (^vvbp)(void);
vvbp f41(void) { }

// CHECK: define i32 @f42()
struct s42 { enum e40 f0; } f42(void) {  }

// CHECK: define i64 @f43()
struct s43 { enum e40 f0; int f1; } f43(void) {  }

// CHECK: define i32 @f44()
struct s44 { vvbp f0; } f44(void) {  }

// CHECK: define i64 @f45()
struct s45 { vvbp f0; int f1; } f45(void) {  }

// CHECK: define void @f46(i32 %a0)
void f46(enum e40 a0) { }

// CHECK: define void @f47(void ()* %a1)
void f47(vvbp a1) { }

// CHECK: define void @f48(i32 %a0.0)
struct s48 { enum e40 f0; };
void f48(struct s48 a0) { }

// CHECK: define void @f49(i32 %a0.0, i32 %a0.1)
struct s49 { enum e40 f0; int f1; };
void f49(struct s49 a0) { }

// CHECK: define void @f50(void ()* %a0.0)
struct s50 { vvbp f0; };
void f50(struct s50 a0) { }

// CHECK: define void @f51(void ()* %a0.0, i32 %a0.1)
struct s51 { vvbp f0; int f1; };
void f51(struct s51 a0) { }

// CHECK: define void @f52(%struct.s52* byval align 16 %x)
struct s52 {
  long double a;
};
void f52(struct s52 x) {}

// CHECK: define void @f53(%struct.s53* byval align 32 %x)
struct __attribute__((aligned(32))) s53 {
  int x;
  int y;
};
void f53(struct s53 x) {}

typedef unsigned short v2i16 __attribute__((__vector_size__(4)));

// CHECK: define i32 @f54(i32 %arg.coerce)
// rdar://8359483
v2i16 f54(v2i16 arg) { return arg+arg; }

