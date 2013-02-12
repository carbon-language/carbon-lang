// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s
// rdar://12470373

// Checks to make sure we parse the binary suffix properly.
// CHECK: movl $1, %eax
  mov eax, 01b
// CHECK: movl $2, %eax
  mov eax, 10b
// CHECK: movl $3, %eax
  mov eax, 11b
// CHECK: movl $3, %eax
  mov eax, 11B
// CHECK: movl $2711, %eax
  mov eax, 101010010111B
