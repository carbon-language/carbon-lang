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
// RUN: grep 'define i64 @f9_1()' %t &&

// FIXME: This is wrong, but we want the coverage of the other
// tests. This should be the same as @f8_2.
// RUN: grep 'define void @f9_2(%.truct.s9\* byval %a0)' %t &&

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

// FIXME: This is currently broken, but the test case is accepting it
// so we get coverage of the other cases.
struct s9 {
  int a : 17;
  int b;
};
struct s9 f9_1(void) {
}
void f9_2(struct s9 a0) {
}

// Return of small structures and unions...

// RUN: grep 'float @f10()' %t &&
struct s10 {
  union { };
  float f;
} f10(void) {}

// Small vectors and 1 x {i64,double} are returned in registers...

// RUN: grep 'i32 @f11()' %t &&
// RUN: grep -F 'void @f12(<2 x i32>* noalias sret %agg.result)' %t &&
// RUN: grep 'i64 @f13()' %t &&
// RUN: grep 'i64 @f14()' %t &&
typedef short T11 __attribute__ ((vector_size (4)));
T11 f11(void) {}
typedef int T12 __attribute__ ((vector_size (8)));
T12 f12(void) {}
typedef long long T13 __attribute__ ((vector_size (8)));
T13 f13(void) {}
typedef double T14 __attribute__ ((vector_size (8)));
T14 f14(void) {}

// RUN: true
