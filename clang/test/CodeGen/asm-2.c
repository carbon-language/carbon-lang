// RUN: clang-cc -emit-llvm %s -o %t -arch=i386 -O2 &&
// RUN: not grep "load" %t

// <rdar://problem/6841383>
int cpuid(unsigned data) {
  int a, b;
  
  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
}