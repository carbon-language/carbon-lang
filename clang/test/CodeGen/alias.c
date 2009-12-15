// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o %t %s
// RUN: grep '@g0 = common global i32 0' %t
// RUN: grep '@f1 = alias void ()\* @f0' %t
// RUN: grep '@g1 = alias i32\* @g0' %t
// RUN: grep 'define void @f0() nounwind {' %t

void f0(void) { }
extern void f1(void);
extern void f1(void) __attribute((alias("f0")));

int g0;
extern int g1;
extern int g1 __attribute((alias("g0")));

// Make sure that aliases cause referenced values to be emitted.
// PR3200
// RUN: grep 'define internal i32 @foo1()' %t
static inline int foo1() { return 0; }
int foo() __attribute__((alias("foo1")));


// RUN: grep '@bar1 = internal global i32 42' %t
static int bar1 = 42;
int bar() __attribute__((alias("bar1")));


extern int test6();
void test7() { test6(); }  // test6 is emitted as extern.

// test6 changes to alias.
int test6() __attribute__((alias("test7")));

