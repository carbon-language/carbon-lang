// RUN: %llvmgxx %s -fasm-blocks -S -o - | FileCheck %s
// Complicated expression as jump target
// XFAIL: *
// XTARGET: x86,i386,i686,darwin

void Method3()
{
// CHECK: Method3
// CHECK-NOT: alignstack
    asm("foo:");
// CHECK: return
}

void Method4()
{
// CHECK: Method4
// CHECK: alignstack
  asm {
    bar:
  }
// CHECK: return
}

