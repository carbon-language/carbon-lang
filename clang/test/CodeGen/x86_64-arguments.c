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
// CHECK: define void @f8_2(i64 %a0.coerce0, double %a0.coerce1)
union u8 {
  long double a;
  int b;
};
union u8 f8_1() { while (1) {} }
void f8_2(union u8 a0) {}

// CHECK: define i64 @f9()
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// CHECK: define void @f10(i64 %a0.coerce)
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// CHECK: define void @f11(%struct.s19* sret %agg.result)
union { long double a; float b; } f11() { while (1) {} }

// CHECK: define i64 @f12_0()
// CHECK: define void @f12_1(i64 %a0.coerce)
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// CHECK: define void @f13(%struct.s13_0* sret %agg.result, i32 %a, i32 %b, i32 %c, i32 %d, {{.*}}* byval %e, i32 %f)

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

// Check for valid coercion.  The struct should be passed/returned as i32, not
// as i64 for better code quality.
// rdar://8135035
// CHECK: define void @f18(i32 %a, i32 %f18_arg1.coerce) 
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

struct StringRef {
  long x;
  const char *Ptr;
};

// rdar://7375902
// CHECK: define i8* @f21(i64 %S.coerce0, i8* %S.coerce1) 
const char *f21(struct StringRef S) { return S.x+S.Ptr; }

// PR7567
typedef __attribute__ ((aligned(16))) struct f22s { unsigned long long x[2]; } L;
void f22(L x, L y) { }
// CHECK: @f22
// CHECK: %x = alloca{{.*}}, align 16
// CHECK: %y = alloca{{.*}}, align 16



// PR7714
struct f23S {
  short f0;
  unsigned f1;
  int f2;
};


void f23(int A, struct f23S B) {
  // CHECK: define void @f23(i32 %A, i64 %B.coerce0, i32 %B.coerce1)
}

struct f24s { long a; int b; };

struct f23S f24(struct f23S *X, struct f24s *P2) {
  return *X;
  
  // CHECK: define %struct.f24s @f24(%struct.f23S* %X, %struct.f24s* %P2)
}

// rdar://8248065
typedef float v4f32 __attribute__((__vector_size__(16)));
v4f32 f25(v4f32 X) {
  // CHECK: define <4 x float> @f25(<4 x float> %X)
  // CHECK-NOT: alloca
  // CHECK: %X.addr = alloca <4 x float>
  // CHECK-NOT: alloca
  // CHECK: store <4 x float> %X, <4 x float>* %X.addr
  // CHECK-NOT: store
  // CHECK: ret <4 x float>
  return X+X;
}

struct foo26 {
  int *X;
  float *Y;
};

struct foo26 f26(struct foo26 *P) {
  // CHECK: define %struct.foo26 @f26(%struct.foo26* %P)
  return *P;
}


struct v4f32wrapper {
  v4f32 v;
};

struct v4f32wrapper f27(struct v4f32wrapper X) {
  // CHECK: define <4 x float> @f27(<4 x float> %X.coerce)
  return X;
}