// RUN: clang -emit-llvm %s -o - | grep _ZplRK1YRA100_P1X

// FIXME: This test is intentionally trivial, because we can't yet
// CodeGen anything real in C++.
struct X { };
struct Y { };
  
bool operator+(const Y&, X* (&xs)[100]) { return false; }

