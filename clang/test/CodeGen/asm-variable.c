// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

unsigned long long foo(unsigned long long addr, unsigned long long a0,
                       unsigned long long a1, unsigned long long a2,
                       unsigned long long a3, unsigned long long a4,
                       unsigned long long a5) {
  register unsigned long long result asm("rax");
  register unsigned long long b0 asm("rdi");
  register unsigned long long b1 asm("rsi");
  register unsigned long long b2 asm("rdx");
  register unsigned long long b3 asm("rcx");
  register unsigned long long b4 asm("r8");
  register unsigned long long b5 asm("r9");

  b0 = a0;
  b1 = a1;
  b2 = a2;
  b3 = a3;
  b4 = a4;
  b5 = a5;

  asm("call *%1" : "=r" (result)
      : "r"(addr), "r" (b0), "r" (b1), "r" (b2), "r" (b3), "r" (b4), "r" (b5));
  return result;
}

// CHECK: call i64 asm "call *$1", "={rax},r,{rdi},{rsi},{rdx},{rcx},{r8},{r9},~{dirflag},~{fpsr},~{flags}"
