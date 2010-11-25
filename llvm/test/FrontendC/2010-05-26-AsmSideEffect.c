// RUN: %llvmgcc %s -S -o - | FileCheck %s
// Radar 8026855

int test (void *src) {
  register int w0 asm ("0");
  // CHECK: call i32 asm sideeffect
  asm ("ldr %0, [%1]": "=r" (w0): "r" (src));
  // The asm to read the value of w0 has a sideeffect for a different reason
  // (see 2010-05-18-asmsched.c) but that's not what this is testing for.
  // CHECK: call i32 asm
  return w0;
}
