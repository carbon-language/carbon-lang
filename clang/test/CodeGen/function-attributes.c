// RUN: clang -emit-llvm -o %t %s &&
// RUN: grep 'define signext i8 @f0(i32 %x) nounwind' %t &&
// RUN: grep 'define zeroext i8 @f1(i32 %x) nounwind' %t &&
// RUN: grep 'define void @f2(i8 signext %x) nounwind' %t &&
// RUN: grep 'define void @f3(i8 zeroext %x) nounwind' %t &&
// RUN: grep 'define signext i16 @f4(i32 %x) nounwind' %t &&
// RUN: grep 'define zeroext i16 @f5(i32 %x) nounwind' %t &&
// RUN: grep 'define void @f6(i16 signext %x) nounwind' %t &&
// RUN: grep 'define void @f7(i16 zeroext %x) nounwind' %t &&
// RUN: grep 'define void @f8() nounwind alwaysinline' %t

signed char f0(int x) { return x; }

unsigned char f1(int x) { return x; }

void f2(signed char x) { }

void f3(unsigned char x) { }

signed short f4(int x) { return x; }

unsigned short f5(int x) { return x; }

void f6(signed short x) { }

void f7(unsigned short x) { }

void __attribute__((always_inline)) f8(void) { }
