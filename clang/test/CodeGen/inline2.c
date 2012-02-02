// RUN: echo "GNU89 tests:"
// RUN: %clang_cc1 -O1 -triple i386-apple-darwin9 -emit-llvm -o %t -std=gnu89 %s
// RUN: grep "define i32 @f0()" %t
// RUN: grep "define i32 @f1()" %t
// RUN: grep "define i32 @f2()" %t
// RUN: grep "define i32 @f3()" %t
// RUN: grep "define i32 @f5()" %t
// RUN: grep "define i32 @f6()" %t
// RUN: grep "define i32 @f7()" %t
// RUN: grep "define i32 @fA()" %t
// RUN: grep "define available_externally i32 @f4()" %t
// RUN: grep "define available_externally i32 @f8()" %t
// RUN: grep "define available_externally i32 @f9()" %t

// RUN: echo "C99 tests:"
// RUN: %clang_cc1 -O1 -triple i386-apple-darwin9 -emit-llvm -o %t -std=c99 %s
// RUN: grep "define i32 @f0()" %t
// RUN: grep "define i32 @f1()" %t
// RUN: grep "define i32 @f2()" %t
// RUN: grep "define i32 @f3()" %t
// RUN: grep "define i32 @f5()" %t
// RUN: grep "define i32 @f6()" %t
// RUN: grep "define i32 @f7()" %t
// RUN: grep "define available_externally i32 @fA()" %t
// RUN: grep "define i32 @f4()" %t
// RUN: grep "define i32 @f8()" %t
// RUN: grep "define i32 @f9()" %t

int f0(void);
int f0(void) { return 0; }

inline int f1(void);
int f1(void) { return 0; }

int f2(void);
inline int f2(void) { return 0; }

extern inline int f3(void);
int f3(void) { return 0; }

extern inline int f5(void);
inline int f5(void) { return 0; }

inline int f6(void);
extern inline int f6(void) { return 0; }

extern inline int f7(void);
extern int f7(void) { return 0; }

inline int fA(void) { return 0; }

int f4(void);
extern inline int f4(void) { return 0; }

extern int f8(void);
extern inline int f8(void) { return 0; }

extern inline int f9(void);
extern inline int f9(void) { return 0; }

int test_all() { 
  return f0() + f1() + f2() + f3() + f4() + f5() + f6() + f7() + f8() + f9() 
    + fA();
}
