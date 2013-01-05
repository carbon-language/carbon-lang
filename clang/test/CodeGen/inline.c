// RUN: echo "GNU89 tests:"
// RUN: %clang %s -target i386-unknown-unknown -O1 -emit-llvm -S -o - -std=gnu89 | FileCheck %s --check-prefix=CHECK1
// CHECK1: define i32 @foo()
// CHECK1: define i32 @bar()
// CHECK1: define void @unreferenced1()
// CHECK1-NOT: unreferenced2
// CHECK1: define void @gnu_inline()
// CHECK1: define i32 @test1
// CHECK1: define i32 @test2
// CHECK1: define void @test3()
// CHECK1: define available_externally i32 @test4
// CHECK1: define available_externally i32 @test5
// CHECK1: define i32 @test6
// CHECK1: define void @test7
// CHECK1: define i{{..}} @strlcpy
// CHECK1-NOT: test9
// CHECK1: define void @testA
// CHECK1: define void @testB
// CHECK1: define void @testC
// CHECK1: define available_externally void @gnu_ei_inline()
// CHECK1: define available_externally i32 @ei()

// RUN: echo "C99 tests:"
// RUN: %clang %s -target i386-unknown-unknown -O1 -emit-llvm -S -o - -std=gnu99 | FileCheck %s --check-prefix=CHECK2
// CHECK2: define i32 @ei()
// CHECK2: define i32 @bar()
// CHECK2-NOT: unreferenced1
// CHECK2: define void @unreferenced2()
// CHECK2: define void @gnu_inline()
// CHECK2: define i32 @test1
// CHECK2: define i32 @test2
// CHECK2: define void @test3
// CHECK2: define available_externally i32 @test4
// CHECK2: define available_externally i32 @test5
// CHECK2: define i32 @test6
// CHECK2: define void @test7
// CHECK2: define available_externally i{{..}} @strlcpy
// CHECK2: define void @test9
// CHECK2: define void @testA
// CHECK2: define void @testB
// CHECK2: define void @testC
// CHECK2: define available_externally void @gnu_ei_inline()
// CHECK2: define available_externally i32 @foo()

// RUN: echo "C++ tests:"
// RUN: %clang -x c++ %s -target i386-unknown-unknown -O1 -emit-llvm -S -o - -std=c++98 | FileCheck %s --check-prefix=CHECK3
// CHECK3: define i32 @_Z3barv()
// CHECK3: define linkonce_odr i32 @_Z3foov()
// CHECK3-NOT: unreferenced
// CHECK3: define void @_Z10gnu_inlinev()
// CHECK3: define available_externally void @_Z13gnu_ei_inlinev()
// CHECK3: define linkonce_odr i32 @_Z2eiv()

extern __inline int ei() { return 123; }

__inline int foo() {
  return ei();
}

int bar() { return foo(); }


__inline void unreferenced1() {}
extern __inline void unreferenced2() {}

__inline __attribute((__gnu_inline__)) void gnu_inline() {}

// PR3988
extern __inline __attribute__((gnu_inline)) void gnu_ei_inline() {}
void (*P)() = gnu_ei_inline;

// <rdar://problem/6818429>
int test1();
__inline int test1() { return 4; }
__inline int test2() { return 5; }
__inline int test2();
int test2();

void test_test1() { test1(); }
void test_test2() { test2(); }

// PR3989
extern __inline void test3() __attribute__((gnu_inline));
__inline void __attribute__((gnu_inline)) test3() {}

extern int test4(void);
extern __inline __attribute__ ((__gnu_inline__)) int test4(void)
{
  return 0;
}

void test_test4() { test4(); }

extern __inline int test5(void)  __attribute__ ((__gnu_inline__));
extern __inline int __attribute__ ((__gnu_inline__)) test5(void)
{
  return 0;
}

void test_test5() { test5(); }

// PR10233

__inline int test6() { return 0; }
extern int test6();


// No PR#, but this once crashed clang in C99 mode due to buggy extern inline
// redeclaration detection.
void test7() { }
void test7();

// PR11062; the fact that the function is named strlcpy matters here.
inline __typeof(sizeof(int)) strlcpy(char *dest, const char *src, __typeof(sizeof(int)) size) { return 3; }
void test8() { strlcpy(0,0,0); }

// PR10657; the test crashed in C99 mode
extern inline void test9() { }
void test9();

inline void testA() {}
void testA();

void testB();
inline void testB() {}
extern void testB();

extern inline void testC() {}
inline void testC();
