// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm -o %t %s &&
// RUN: grep 'define signext i8 @f0(i32 %x) nounwind' %t &&
// RUN: grep 'define zeroext i8 @f1(i32 %x) nounwind' %t &&
// RUN: grep 'define void @f2(i8 signext %x) nounwind' %t &&
// RUN: grep 'define void @f3(i8 zeroext %x) nounwind' %t &&
// RUN: grep 'define signext i16 @f4(i32 %x) nounwind' %t &&
// RUN: grep 'define zeroext i16 @f5(i32 %x) nounwind' %t &&
// RUN: grep 'define void @f6(i16 signext %x) nounwind' %t &&
// RUN: grep 'define void @f7(i16 zeroext %x) nounwind' %t &&

signed char f0(int x) { return x; }

unsigned char f1(int x) { return x; }

void f2(signed char x) { }

void f3(unsigned char x) { }

signed short f4(int x) { return x; }

unsigned short f5(int x) { return x; }

void f6(signed short x) { }

void f7(unsigned short x) { }

// F8 is dead so it should not be emitted.
// RUN: not grep '@f8' %t &&
void __attribute__((always_inline)) f8(void) { }

// RUN: grep 'call void @f9_t() noreturn' %t &&
void __attribute__((noreturn)) f9_t(void);
void f9(void) { f9_t(); }

// FIXME: We should be setting nounwind on calls.
// RUN: grep 'call i32 @f10_t() readnone' %t &&
int __attribute__((const)) f10_t(void);
int f10(void) { return f10_t(); }
int f11(void) {
 exit:
  return f10_t();
}
int f12(int arg) {
  return arg ? 0 : f10_t();
}

// RUN: grep 'define void @f13() nounwind readnone' %t &&
void f13(void) __attribute__((pure)) __attribute__((const));
void f13(void){}

// RUN: true
