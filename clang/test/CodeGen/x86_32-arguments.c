// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm -o %t %s &&
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
}

short f1(void) {
}

int f2(void) {
}

float f3(void) {
}

double f4(void) {
}

long double f5(void) {
}

void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

typedef enum { A, B, C } E;

void f7(E a0) {
}

struct s8 {
  int a;
  int b;
};
struct s8 f8_1(void) {
}
void f8_2(struct s8 a0) {
}

// This should be passed just as s8.

// RUN: grep 'define i64 @f9_1()' %t &&

// FIXME: llvm-gcc expands this, this may have some value for the
// backend in terms of optimization but doesn't change the ABI.
// RUN: grep 'define void @f9_2(%.truct.s9\* byval %a0)' %t &&
struct s9 {
  int a : 17;
  int b;
};
struct s9 f9_1(void) {
}
void f9_2(struct s9 a0) {
}

// Return of small structures and unions

// RUN: grep 'float @f10()' %t &&
struct s10 {
  union { };
  float f;
} f10(void) {}

// Small vectors and 1 x {i64,double} are returned in registers

// RUN: grep 'i32 @f11()' %t &&
// RUN: grep -F 'void @f12(<2 x i32>* noalias sret %agg.result)' %t &&
// RUN: grep 'i64 @f13()' %t &&
// RUN: grep 'i64 @f14()' %t &&
// RUN: grep '<2 x i64> @f15()' %t &&
// RUN: grep '<2 x i64> @f16()' %t &&
typedef short T11 __attribute__ ((vector_size (4)));
T11 f11(void) {}
typedef int T12 __attribute__ ((vector_size (8)));
T12 f12(void) {}
typedef long long T13 __attribute__ ((vector_size (8)));
T13 f13(void) {}
typedef double T14 __attribute__ ((vector_size (8)));
T14 f14(void) {}
typedef long long T15 __attribute__ ((vector_size (16)));
T15 f15(void) {}
typedef double T16 __attribute__ ((vector_size (16)));
T16 f16(void) {}

// And when the single element in a struct (but not for 64 and
// 128-bits).

// RUN: grep 'i32 @f17()' %t &&
// RUN: grep -F 'void @f18(%2* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f19(%3* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f20(%4* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f21(%5* noalias sret %agg.result)' %t &&
// RUN: grep -F 'void @f22(%6* noalias sret %agg.result)' %t &&
struct { T11 a; } f17(void) {}
struct { T12 a; } f18(void) {}
struct { T13 a; } f19(void) {}
struct { T14 a; } f20(void) {}
struct { T15 a; } f21(void) {}
struct { T16 a; } f22(void) {}

// Single element structures are handled specially

// RUN: grep -F 'float @f23()' %t &&
// RUN: grep -F 'float @f24()' %t &&
// RUN: grep -F 'float @f25()' %t &&
struct { float a; } f23(void) {}
struct { float a[1]; } f24(void) {}
struct { struct {} a; struct { float a[1]; } b; } f25(void) {}

// Small structures are handled recursively
// RUN: grep -F 'i32 @f26()' %t &&
// RUN: grep 'void @f27(%.truct.s27\* noalias sret %agg.result)' %t &&
struct s26 { struct { char a, b; } a; struct { char a, b; } b; } f26(void) {}
struct s27 { struct { char a, b, c; } a; struct { char a; } b; } f27(void) {}

// RUN: grep 'void @f28(%.truct.s28\* noalias sret %agg.result)' %t &&
struct s28 { int a; int b[]; } f28(void) {}

// RUN: grep 'define i16 @f29()' %t &&
struct s29 { struct { } a[1]; char b; char c; } f29(void) {}

// RUN: grep 'define i16 @f30()' %t &&
struct s30 { char a; char b : 4; } f30(void) {}

// RUN: grep 'define float @f31()' %t &&
struct s31 { char : 0; float b; char : 0; } f31(void) {}

// RUN: grep 'define i32 @f32()' %t &&
struct s32 { char a; unsigned : 0; } f32(void) {}

// RUN: grep 'define float @f33()' %t &&
struct s33 { float a; long long : 0; } f33(void) {}

// RUN: grep 'define float @f34()' %t &&
struct s34 { struct { int : 0; } a; float b; } f34(void) {}

// RUN: grep 'define i16 @f35()' %t &&
struct s35 { struct { int : 0; } a; char b; char c; } f35(void) {}

// RUN: grep 'define i16 @f36()' %t &&
struct s36 { struct { int : 0; } a[2][10]; char b; char c; } f36(void) {}

// RUN: grep 'define float @f37()' %t &&
struct s37 { float c[1][1]; } f37(void) {}

// RUN: grep 'define void @f38(.struct.s38. noalias sret .agg.result)' %t &&
struct s38 { char a[3]; short b; } f38(void) {}

// RUN: true
