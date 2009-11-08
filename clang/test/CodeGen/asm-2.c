// RUN: clang-cc -emit-llvm %s -o %t -triple i386-pc-linux-gnu -O2
// RUN: not grep "load" %t

// <rdar://problem/6841383>
int cpuid(unsigned data) {
  int a, b;
  
  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
}
