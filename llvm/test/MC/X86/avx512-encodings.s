// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl --show-encoding %s | FileCheck %s

// CHECK: vmovdqu64 {{.*}} {%k3}
// CHECK: encoding: [0x62,0xf1,0xfe,0x4b,0x6f,0xc8]
vmovdqu64 %zmm0, %zmm1 {%k3}
