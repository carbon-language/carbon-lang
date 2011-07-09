// RUN: %llvmgcc -mmmx -S -o - %s | FileCheck %s
// XFAIL: *
// XTARGET: x86,i386,i686
// <rdar://problem/9091220>
#include <mmintrin.h>
#include <stdint.h>

// CHECK: { x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx }

void foo(__m64 vfill) {
  __m64 v1, v2, v3, v4, v5, v6, v7;

  __asm__ __volatile__ (
    "\tmovq  %7, %0\n"
    "\tmovq  %7, %1\n"
    "\tmovq  %7, %2\n"
    "\tmovq  %7, %3\n"
    "\tmovq  %7, %4\n"
    "\tmovq  %7, %5\n"
    "\tmovq  %7, %6"
    : "=&y" (v1), "=&y" (v2), "=&y" (v3),
      "=&y" (v4), "=&y" (v5), "=&y" (v6), "=y" (v7)
    : "y" (vfill));
}
