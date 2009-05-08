// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o %t %s &&
// RUN: grep 'define signext i8 @f0()' %t &&
// RUN: grep 'define signext i16 @f1()' %t &&
// RUN: grep 'define i32 @f2()' %t &&
// RUN: grep 'define float @f3()' %t &&
// RUN: grep 'define double @f4()' %t &&
// RUN: grep 'define x86_fp80 @f5()' %t &&
// RUN: grep 'define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8\* %a4)' %t &&
// RUN: grep 'define void @f7(i32 %a0)' %t &&
// RUN: grep 'type { i64, double }.*type .0' %t &&
// RUN: grep 'define .0 @f8_1()' %t &&
// RUN: grep 'define void @f8_2(.0)' %t &&

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

// Test merging/passing of upper eightbyte with X87 class.
union u8 {
  long double a;
  int b;
};
union u8 f8_1() {}
void f8_2(union u8 a0) {}

// RUN: grep 'define i64 @f9()' %t &&
struct s9 { int a; int b; int : 0; } f9(void) {}

// RUN: grep 'define void @f10(i64)' %t &&
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// RUN: true
