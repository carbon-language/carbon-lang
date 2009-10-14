// RUN: %llvmgcc %s -fasm-blocks -S -o - | FileCheck %s
// Complicated expression as jump target
// XFAIL: *
// XTARGET: x86,i386,i686

void Method3()
{
// CHECK: Method3
// CHECK-NOT: msasm
    asm("foo:");
// CHECK: return
}

void Method4()
{
// CHECK: Method4
// CHECK: msasm
  asm {
    bar:
  }
// CHECK: return
}

