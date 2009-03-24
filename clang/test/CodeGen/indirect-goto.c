// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t &&
// RUN: grep "ret i32" %t | count 1 &&
// RUN: grep "ret i32 210" %t | count 1

static int foo(unsigned i) {
  const void *addrs[] = { &&L1, &&L2, &&L3, &&L4, &&L5 };
  int res = 1;

  goto *addrs[i];
 L5: res *= 11;
 L4: res *= 7;
 L3: res *= 5;
 L2: res *= 3;
 L1: res *= 2; 
  return res;
}

int bar() {
  return foo(3);
}
