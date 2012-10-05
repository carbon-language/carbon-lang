// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

int g0;
// CHECK: @g0 = common global i32 0
static int bar1 = 42;
// CHECK: @bar1 = internal global i32 42

extern int g1;
extern int g1 __attribute((alias("g0")));
// CHECK: @g1 = alias i32* @g0

void f0(void) { }
extern void f1(void);
extern void f1(void) __attribute((alias("f0")));
// CHECK: @f1 = alias void ()* @f0
// CHECK: define void @f0() nounwind {

// Make sure that aliases cause referenced values to be emitted.
// PR3200
static inline int foo1() { return 0; }
// CHECK: define internal i32 @foo1()
int foo() __attribute__((alias("foo1")));
int bar() __attribute__((alias("bar1")));

extern int test6();
void test7() { test6(); }  // test6 is emitted as extern.

// test6 changes to alias.
int test6() __attribute__((alias("test7")));

