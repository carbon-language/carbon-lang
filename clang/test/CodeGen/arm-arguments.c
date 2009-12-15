// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=APCS-GNU %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi aapcs -emit-llvm -w -o - %s | FileCheck -check-prefix=AAPCS %s

// APCS-GNU: define arm_apcscc signext i8 @f0()
// AAPCS: define arm_aapcscc signext i8 @f0()
char f0(void) {
  return 0;
}

// APCS-GNU: define arm_apcscc i8 @f1()
// AAPCS: define arm_aapcscc i8 @f1()
struct s1 { char f0; };
struct s1 f1(void) {}

// APCS-GNU: define arm_apcscc i16 @f2()
// AAPCS: define arm_aapcscc i16 @f2()
struct s2 { short f0; };
struct s2 f2(void) {}

// APCS-GNU: define arm_apcscc i32 @f3()
// AAPCS: define arm_aapcscc i32 @f3()
struct s3 { int f0; };
struct s3 f3(void) {}

// APCS-GNU: define arm_apcscc i32 @f4()
// AAPCS: define arm_aapcscc i32 @f4()
struct s4 { struct s4_0 { int f0; } f0; };
struct s4 f4(void) {}

// APCS-GNU: define arm_apcscc void @f5(
// APCS-GNU: struct.s5* noalias sret
// AAPCS: define arm_aapcscc i32 @f5()
struct s5 { struct { } f0; int f1; };
struct s5 f5(void) {}

// APCS-GNU: define arm_apcscc void @f6(
// APCS-GNU: struct.s6* noalias sret
// AAPCS: define arm_aapcscc i32 @f6()
struct s6 { int f0[1]; };
struct s6 f6(void) {}

// APCS-GNU: define arm_apcscc void @f7()
// AAPCS: define arm_aapcscc void @f7()
struct s7 { struct { int : 0; } f0; };
struct s7 f7(void) {}

// APCS-GNU: define arm_apcscc void @f8(
// APCS-GNU: struct.s8* noalias sret
// AAPCS: define arm_aapcscc void @f8()
struct s8 { struct { int : 0; } f0[1]; };
struct s8 f8(void) {}

// APCS-GNU: define arm_apcscc i32 @f9()
// AAPCS: define arm_aapcscc i32 @f9()
struct s9 { int f0; int : 0; };
struct s9 f9(void) {}

// APCS-GNU: define arm_apcscc i32 @f10()
// AAPCS: define arm_aapcscc i32 @f10()
struct s10 { int f0; int : 0; int : 0; };
struct s10 f10(void) {}

// APCS-GNU: define arm_apcscc void @f11(
// APCS-GNU: struct.s10* noalias sret
// AAPCS: define arm_aapcscc i32 @f11()
struct s11 { int : 0; int f0; };
struct s11 f11(void) {}

// APCS-GNU: define arm_apcscc i32 @f12()
// AAPCS: define arm_aapcscc i32 @f12()
union u12 { char f0; short f1; int f2; };
union u12 f12(void) {}

// APCS-GNU: define arm_apcscc void @f13(
// APCS-GNU: struct.s13* noalias sret

// FIXME: This should return a float.
// AAPCS-FIXME: define arm_aapcscc float @f13()
struct s13 { float f0; };
struct s13 f13(void) {}

// APCS-GNU: define arm_apcscc void @f14(
// APCS-GNU: struct.s13* noalias sret
// AAPCS: define arm_aapcscc i32 @f14()
union u14 { float f0; };
union u14 f14(void) {}

// APCS-GNU: define arm_apcscc void @f15()
// AAPCS: define arm_aapcscc void @f15()
void f15(struct s7 a0) {}

// APCS-GNU: define arm_apcscc void @f16()
// AAPCS: define arm_aapcscc void @f16()
void f16(struct s8 a0) {}
