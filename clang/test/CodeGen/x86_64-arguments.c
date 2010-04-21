// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK: %0 = type { i64, double }

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
void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

// CHECK: define void @f7(i32 %a0)
typedef enum { A, B, C } e7;
void f7(e7 a0) {
}

// Test merging/passing of upper eightbyte with X87 class.
//
// CHECK: define %0 @f8_1()
// CHECK: define void @f8_2(%0)
union u8 {
  long double a;
  int b;
};
union u8 f8_1() { while (1) {} }
void f8_2(union u8 a0) {}

// CHECK: define i64 @f9()
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// CHECK: define void @f10(i64)
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// CHECK: define void @f11(%struct.s19* sret %agg.result)
union { long double a; float b; } f11() { while (1) {} }

// CHECK: define i64 @f12_0()
// CHECK: define void @f12_1(i64)
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// CHECK: define void @f13(%struct.s13_0* sret %agg.result, i32 %a, i32 %b, i32 %c, i32 %d, %struct.s13_1* byval %e, i32 %f)

struct s13_0 { long long f0[3]; };
struct s13_1 { long long f0[2]; };
struct s13_0 f13(int a, int b, int c, int d,
                 struct s13_1 e, int f) { while (1) {} }

// CHECK: define void @f14({{.*}}, i8 signext %X)
void f14(int a, int b, int c, int d, int e, int f, char X) {}

// CHECK: define void @f15({{.*}}, i8* %X)
void f15(int a, int b, int c, int d, int e, int f, void *X) {}

// CHECK: define void @f16({{.*}}, float %X)
void f16(float a, float b, float c, float d, float e, float f, float g, float h,
         float X) {}

// CHECK: define void @f17({{.*}}, x86_fp80 %X)
void f17(float a, float b, float c, float d, float e, float f, float g, float h,
         long double X) {}

// Check for valid coercion.
// CHECK: [[f18_t0:%.*]] = bitcast i64* {{.*}} to %struct.f18_s0*
// CHECK: [[f18_t1:%.*]] = load %struct.f18_s0* [[f18_t0]], align 1
// CHECK: store %struct.f18_s0 [[f18_t1]], %struct.f18_s0* %f18_arg1
struct f18_s0 { int f0; };
void f18(int a, struct f18_s0 f18_arg1) { while (1) {} }

// Check byval alignment.

// CHECK: define void @f19(%struct.s19* byval align 16 %x)
struct s19 {
  long double a;
};
void f19(struct s19 x) {}

// CHECK: define void @f20(%struct.s20* byval align 32 %x)
struct __attribute__((aligned(32))) s20 {
  int x;
  int y;
};
void f20(struct s20 x) {}
