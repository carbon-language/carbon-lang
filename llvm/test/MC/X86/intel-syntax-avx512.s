// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -mcpu=knl --show-encoding %s | FileCheck %s

// CHECK: vaddps (%rax), %zmm1, %zmm1
// CHECK: encoding: [0x62,0xf1,0x74,0x48,0x58,0x08]
vaddps zmm1, zmm1, zmmword ptr [rax]
