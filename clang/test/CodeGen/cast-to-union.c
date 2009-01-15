// RUN: clang -emit-llvm < %s | grep "store i32 351, i32*"

union u { int i; };

void foo() {
  union u ola = (union u) 351;
}

// FIXME: not working yet
// union u w = (union u)2;
