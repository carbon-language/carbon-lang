// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=PCS %s

// Sign extension is performed by the callee on AArch64, which means
// that we *shouldn't* tag arguments and returns with their extension.

// PCS: define i8 @f0(i16 %a)
char f0(short a) {
  return a;
}

// PCS: define [1 x i64] @f1()
struct s1 { char f0; };
struct s1 f1(void) {}

// PCS: define [1 x i64] @f2()
struct s2 { short f0; };
struct s2 f2(void) {}

// PCS: define [1 x i64] @f3()
struct s3 { int f0; };
struct s3 f3(void) {}

// PCS: define [1 x i64] @f4()
struct s4 { struct s4_0 { int f0; } f0; };
struct s4 f4(void) {}

// PCS: define [1 x i64] @f5()
struct s5 { struct { } f0; int f1; };
struct s5 f5(void) {}

// PCS: define  [1 x i64] @f6()
struct s6 { int f0[1]; };
struct s6 f6(void) {}

// PCS: define void @f7()
struct s7 { struct { int : 0; } f0; };
struct s7 f7(void) {}

// PCS: define  void @f8()
struct s8 { struct { int : 0; } f0[1]; };
struct s8 f8(void) {}

// PCS: define [1 x i64] @f9()
struct s9 { long f0; int : 0; };
struct s9 f9(void) {}

// PCS: define [1 x i64] @f10()
struct s10 { long f0; int : 0; int : 0; };
struct s10 f10(void) {}

// PCS: define [1 x i64] @f11()
struct s11 { int : 0; long f0; };
struct s11 f11(void) {}

// PCS: define [1 x i64] @f12()
union u12 { char f0; short f1; int f2; long f3; };
union u12 f12(void) {}

// PCS: define %struct.s13 @f13()
struct s13 { float f0; };
struct s13 f13(void) {}

// PCS: define %union.u14 @f14()
union u14 { float f0; };
union u14 f14(void) {}

// PCS: define void @f15()
void f15(struct s7 a0) {}

// PCS: define void @f16()
void f16(struct s8 a0) {}

// PCS: define [1 x i64] @f17()
struct s17 { short f0 : 13; char f1 : 4; };
struct s17 f17(void) {}

// PCS: define [1 x i64] @f18()
struct s18 { short f0; char f1 : 4; };
struct s18 f18(void) {}

// PCS: define [1 x i64] @f19()
struct s19 { long f0; struct s8 f1; };
struct s19 f19(void) {}

// PCS: define [1 x i64] @f20()
struct s20 { struct s8 f1; long f0; };
struct s20 f20(void) {}

// PCS: define [1 x i64] @f21()
struct s21 { struct {} f1; long f0 : 4; };
struct s21 f21(void) {}

// PCS: define { float, float } @f22()
// PCS: define { double, double } @f23(
_Complex float      f22(void) {}
_Complex double     f23(void) {}

// PCS: define [1 x i64] @f24()
struct s24 { _Complex char f0; };
struct s24 f24() {}

// PCS: define [1 x i64] @f25()
struct s25 { _Complex short f0; };
struct s25 f25() {}

// PCS: define [1 x i64] @f26()
struct s26 { _Complex int f0; };
struct s26 f26() {}

// PCS: define [2 x i64] @f27()
struct s27 { _Complex long f0; };
struct s27 f27() {}

// PCS: define void @f28(i8 %a, i16 %b, i32 %c, i64 %d, float %e, double %f)
void f28(char a, short b, int c, long d, float e, double f) {}

// PCS: define void @f29([2 x i64] %a
struct s29 { int arr[4]; };
void f29(struct s29 a) {}

// PCS: define void @f30(%struct.s30* %a)
struct s30 { int arr[4]; char c;};
void f30(struct s30 a) {}

// PCS: define void @f31([4 x double] %a
struct s31 { double arr[4]; };
void f31(struct s31 a) {}

// PCS: define void @f32(%struct.s32* %a)
struct s32 { float arr[5]; };
void f32(struct s32 a) {}

// Not the only solution, but it *is* an HFA.
// PCS: define void @f33([3 x float] %a.coerce0, float %a.coerce1)
struct s33 { float arr[3]; float a; };
void f33(struct s33 a) {}

// PCS: define void @f34(%struct.s34* noalias sret
struct s34 { int a[4]; char b };
struct s34 f34(void) {}

// PCS: define void @f35()
struct s35 {};
void f35(struct s35 a) {}

// Check padding is added:
// PCS: @f36(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6, [1 x i64], %struct.s36* byval align 8 %stacked)
struct s36 { long a, b; };
void f36(int x0, int x1, int x2, int x3, int x4, int x5, int x6, struct s36 stacked) {}

// But only once:
// PCS: @f37(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6, [1 x i64], %struct.s37* byval align 8 %stacked, %struct.s37* byval align 8 %stacked2)
struct s37 { long a, b; };
void f37(int x0, int x1, int x2, int x3, int x4, int x5, int x6, struct s37 stacked, struct s37 stacked2) {}

// Check for HFA padding args. Also, they should not end up on the stack in a
// way which will have holes in when lowered further by LLVM. In particular [3 x
// float] would be unacceptable.

// PCS: @f38(float %s0, double %d1, float %s2, float %s3, float %s4, float %s5, [2 x float], %struct.s38* byval align 4 %stacked)
struct s38 { float a, b, c; };
void f38(float s0, double d1, float s2, float s3, float s4, float s5, struct s38 stacked) {}

// Check both VFP and integer arguments are padded (also that pointers and enums
// get counted as integer types correctly).
struct s39_int { long a, b; };
struct s39_float { float a, b, c, d; };
enum s39_enum { Val1, Val2 };
// PCS: @f39(float %s0, i32 %x0, float %s1, i32* %x1, float %s2, i32 %x2, float %s3, float %s4, i32 %x3, [3 x float], %struct.s39_float* byval align 4 %stacked, i32 %x4, i32 %x5, i32 %x6, [1 x i64], %struct.s39_int* byval align 8 %stacked2)
void f39(float s0, int x0, float s1, int *x1, float s2, enum s39_enum x2, float s3, float s4,
         int x3, struct s39_float stacked, int x4, int x5, int x6,
         struct s39_int stacked2) {}

struct s40 { __int128 a; };
// PCS: @f40(i32 %x0, [1 x i128] %x2_3.coerce, i32 %x4, i32 %x5, i32 %x6, [1 x i64], %struct.s40* byval align 16 %stacked)
void f40(int x0, struct s40 x2_3, int x4, int x5, int x6, struct s40 stacked) {}

// Checking: __int128 will get properly aligned type, with padding so big struct doesn't use x7.
struct s41 { int arr[5]; };
// PCS: @f41(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6, [1 x i64], i128* byval align 16, %struct.s41* %stacked2)
int f41(int x0, int x1, int x2, int x3, int x4, int x5, int x6, __int128 stacked, struct s41 stacked2) {}

// Checking: __int128 needing to be aligned in registers will consume correct
// number. Previously padding was inserted before "stacked" because x6_7 was
// "allocated" to x5 and x6 by clang.
// PCS: @f42(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i128 %x6_7, i128* byval align 16)
void f42(int x0, int x1, int x2, int x3, int x4, __int128 x6_7, __int128 stacked) {}

// Checking: __fp16 is extended to double when calling variadic functions
void variadic(int a, ...);
void f43(__fp16 *in) {
  variadic(42, *in);
// CHECK: call void @variadic(i32 42, double
}
