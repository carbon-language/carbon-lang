// RUN: clang -triple i386-pc-linux-gnu -emit-llvm -o %t %s &&
// RUN: grep '@g0 = common global i32 0' %t &&
// RUN: grep '@f1 = alias void ()\* @f0' %t &&
// RUN: grep '@g1 = alias i32\* @g0' %t &&
// RUN: grep 'define void @f0() nounwind {' %t

void f0(void) { }
extern void f1(void);
extern void f1(void) __attribute((alias("f0")));

int g0;
extern int g1;
extern int g1 __attribute((alias("g0")));
