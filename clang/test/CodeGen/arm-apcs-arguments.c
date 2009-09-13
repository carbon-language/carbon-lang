// RUX: iphone-llvm-gcc -arch armv7 -flto -S -o - %s | FileCheck %s
// RUN: clang-cc -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

// CHECK: define arm_apcscc signext i8 @f0()
char f0(void) {
  return 0;
}

// CHECK: define arm_apcscc i8 @f1()
struct s1 { char f0; };
struct s1 f1(void) {}

// CHECK: define arm_apcscc i16 @f2()
struct s2 { short f0; };
struct s2 f2(void) {}

// CHECK: define arm_apcscc i32 @f3()
struct s3 { int f0; };
struct s3 f3(void) {}

// CHECK: define arm_apcscc i32 @f4()
struct s4 { struct s4_0 { int f0; } f0; };
struct s4 f4(void) {}

// CHECK: define arm_apcscc void @f5(
// CHECK: struct.s5* noalias sret
struct s5 { struct { } f0; int f1; };
struct s5 f5(void) {}

// CHECK: define arm_apcscc void @f6(
// CHECK: struct.s6* noalias sret
struct s6 { int f0[1]; };
struct s6 f6(void) {}

// CHECK: define arm_apcscc void @f7()
struct s7 { struct { int : 0; } f0; };
struct s7 f7(void) {}

// CHECK: define arm_apcscc void @f8(
// CHECK: struct.s8* noalias sret
struct s8 { struct { int : 0; } f0[1]; };
struct s8 f8(void) {}

// CHECK: define arm_apcscc i32 @f9()
struct s9 { int f0; int : 0; };
struct s9 f9(void) {}

// CHECK: define arm_apcscc i32 @f10()
struct s10 { int f0; int : 0; int : 0; };
struct s10 f10(void) {}

// CHECK: define arm_apcscc void @f11(
// CHECK: struct.s10* noalias sret
struct s11 { int : 0; int f0; };
struct s11 f11(void) {}

// CHECK: define arm_apcscc i32 @f12()
union u12 { char f0; short f1; int f2; };
union u12 f12(void) {}
