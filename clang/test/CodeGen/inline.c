// RUN: clang %s -emit-llvm -S -o %t -std=c89 &&
// RUN: grep "define available_externally i32 @ei()" %t &&
// RUN: grep "define i32 @foo()" %t &&
// RUN: grep "define i32 @bar()" %t &&
// RUN: grep "define void @unreferenced1()" %t &&
// RUN: not grep unreferenced2 %t &&

// RUN: clang %s -emit-llvm -S -o %t -std=c99 &&
// RUN: grep "define available_externally i32 @ei()" %t &&
// RUN: grep "define available_externally i32 @foo()" %t &&
// RUN: grep "define i32 @bar()" %t &&
// RUN: not grep unreferenced %t &&

// RUN: clang %s -emit-llvm -S -o %t -std=c++98 &&
// RUN: grep "define available_externally i32 @_Z2eiv()" %t &&
// RUN: grep "define linkonce_odr i32 @_Z3foov()" %t &&
// RUN: grep "define i32 @_Z3barv()" %t &&
// RUN: not grep unreferenced %t

extern inline int ei() { return 123; }

inline int foo() {
  return ei();
}

int bar() { return foo(); }


inline void unreferenced1() {}
extern inline void unreferenced2() {}

