// RUN: clang-cc -fblocks -triple i386-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: grep 'define signext i8 @f0()' %t &&
// RUN: grep 'define signext i16 @f1()' %t &&
// RUN: grep 'define i32 @f2()' %t &&
// RUN: grep 'define float @f3()' %t &&
// RUN: grep 'define double @f4()' %t &&
// RUN: grep 'define x86_fp80 @f5()' %t &&
// RUN: grep 'define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8\* %a4)' %t &&
// RUN: grep 'define void @f7(i32 %a0)' %t &&
// RUN: grep 'define i64 @f8_1()' %t && 
// RUN: grep 'define void @f8_2(i32 %a0.0, i32 %a0.1)' %t &&

char f0(void) {
  return 0;
}

short f1(void) {
  return 0;
}

int f2(void) {
  return 0;
}

float f3(void) {
  return 0;
}

double f4(void) {
  return 0;
}

long double f5(void) {
  return 0;
}

void f6(char a0, short a1, int a2, long long a3, void *a4) {}

typedef enum { A, B, C } E;

void f7(E a0) {}

struct s8 {
  int a;
  int b;
};
struct s8 f8_1(void) { while (1) {} }
void f8_2(struct s8 a0) {}

// This should be passed just as s8.

// RUN: grep 'define i64 @f9_1()' %t &&

// FIXME: llvm-gcc expands this, this may have some value for the
// backend in terms of optimization but doesn't change the ABI.
// RUN: grep 'define void @f9_2(%.truct.s9\* byval %a0)' %t &&
struct s9 {
  int a : 17;
  int b;
};
struct s9 f9_1(void) { while (1) {} }
void f9_2(struct s9 a0) {}

// Return of small structures and unions

// RUN: grep 'float @f10()' %t &&
struct s10 {
  union { };
  float f;
} f10(void) { while (1) {} }

// Small vectors and 1 x {i64,double} are returned in registers

// RUN: grep 'i32 @f11()' %t &&
// RUN: grep -F 'void @f12(<2 x i32>* noalias sret %agg.result)' %t &&
// RUN: grep 'i64 @f13()' %t &&
// RUN: grep 'i64 @f14()' %t &&
// RUN: grep '<2 x i64> @f15()' %t &&
// RUN: grep '<2 x i64> @f16()' %t &&
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

// RUN: grep 'i32 @f17()' %t &&
// RUN: grep -F 'void @f18(%2* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f19(%3* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f20(%4* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f21(%5* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f22(%6* noalias sret %agg.result)' %t &&
struct { T11 a; } f17(void) { while (1) {} }
struct { T12 a; } f18(void) { while (1) {} }
struct { T13 a; } f19(void) { while (1) {} }
struct { T14 a; } f20(void) { while (1) {} }
struct { T15 a; } f21(void) { while (1) {} }
struct { T16 a; } f22(void) { while (1) {} }

// Single element structures are handled specially

// RUN: grep -F 'float @f23()' %t &&
// RUN: grep -F 'float @f24()' %t &&
// RUN: grep -F 'float @f25()' %t &&
struct { float a; } f23(void) { while (1) {} }
struct { float a[1]; } f24(void) { while (1) {} }
struct { struct {} a; struct { float a[1]; } b; } f25(void) { while (1) {} }

// Small structures are handled recursively
// RUN: grep -F 'i32 @f26()' %t &&
// RUN: grep 'void @f27(%.truct.s27\* noalias sret %agg.result)' %t &&
struct s26 { struct { char a, b; } a; struct { char a, b; } b; } f26(void) { while (1) {} }
struct s27 { struct { char a, b, c; } a; struct { char a; } b; } f27(void) { while (1) {} }

// RUN: grep 'void @f28(%.truct.s28\* noalias sret %agg.result)' %t &&
struct s28 { int a; int b[]; } f28(void) { while (1) {} }

// RUN: grep 'define i16 @f29()' %t &&
struct s29 { struct { } a[1]; char b; char c; } f29(void) { while (1) {} }

// RUN: grep 'define i16 @f30()' %t &&
struct s30 { char a; char b : 4; } f30(void) { while (1) {} }

// RUN: grep 'define float @f31()' %t &&
struct s31 { char : 0; float b; char : 0; } f31(void) { while (1) {} }

// RUN: grep 'define i32 @f32()' %t &&
struct s32 { char a; unsigned : 0; } f32(void) { while (1) {} }

// RUN: grep 'define float @f33()' %t &&
struct s33 { float a; long long : 0; } f33(void) { while (1) {} }

// RUN: grep 'define float @f34()' %t &&
struct s34 { struct { int : 0; } a; float b; } f34(void) { while (1) {} }

// RUN: grep 'define i16 @f35()' %t &&
struct s35 { struct { int : 0; } a; char b; char c; } f35(void) { while (1) {} }

// RUN: grep 'define i16 @f36()' %t &&
struct s36 { struct { int : 0; } a[2][10]; char b; char c; } f36(void) { while (1) {} }

// RUN: grep 'define float @f37()' %t &&
struct s37 { float c[1][1]; } f37(void) { while (1) {} }

// RUN: grep 'define void @f38(.struct.s38. noalias sret .agg.result)' %t &&
struct s38 { char a[3]; short b; } f38(void) { while (1) {} }

// RUN: grep 'define void @f39(.struct.s39. byval align 16 .x)' %t &&
typedef int v39 __attribute((vector_size(16)));
struct s39 { v39 x; };
void f39(struct s39 x) {}

// <rdar://problem/7247671>
// RUN: grep 'define i32 @f40()' %t &&
enum e40 { ec0 = 0 };
enum e40 f40(void) { }

// RUN: grep 'define void ()\* @f41()' %t &&
typedef void (^vvbp)(void);
vvbp f41(void) { }

// RUN: grep 'define i32 @f42()' %t &&
struct s42 { enum e40 f0; } f42(void) {  }

// RUN: grep 'define i64 @f43()' %t &&
struct s43 { enum e40 f0; int f1; } f43(void) {  }

// RUN: grep 'define i32 @f44()' %t &&
struct s44 { vvbp f0; } f44(void) {  }

// RUN: grep 'define i64 @f45()' %t &&
struct s45 { vvbp f0; int f1; } f45(void) {  }

// RUN: grep 'define void @f46(i32 %a0)' %t &&
void f46(enum e40 a0) { }

// RUN: grep 'define void @f47(void ()\* %a1)' %t &&
void f47(vvbp a1) { }

// RUN: grep 'define void @f48(i32 %a0.0)' %t &&
struct s48 { enum e40 f0; };
void f48(struct s48 a0) { }

// RUN: grep 'define void @f49(i32 %a0.0, i32 %a0.1)' %t &&
struct s49 { enum e40 f0; int f1; };
void f49(struct s49 a0) { }

// RUN: grep 'define void @f50(void ()\* %a0.0)' %t &&
struct s50 { vvbp f0; };
void f50(struct s50 a0) { }

// RUN: grep 'define void @f51(void ()\* %a0.0, i32 %a0.1)' %t &&
struct s51 { vvbp f0; int f1; };
void f51(struct s51 a0) { }

// RUN: true
