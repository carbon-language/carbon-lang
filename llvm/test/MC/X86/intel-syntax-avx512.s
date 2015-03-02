// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -mcpu=knl --show-encoding %s | FileCheck %s

// CHECK: vaddps (%rax), %zmm1, %zmm1
// CHECK: encoding: [0x62,0xf1,0x74,0x48,0x58,0x08]
vaddps zmm1, zmm1, zmmword ptr [rax]

// CHECK: vaddpd  %zmm2, %zmm1, %zmm1
// CHECK:  encoding: [0x62,0xf1,0xf5,0x48,0x58,0xca]
vaddpd zmm1,zmm1,zmm2

// CHECK: vaddpd  %zmm2, %zmm1, %zmm1 {%k5}
// CHECK:  encoding: [0x62,0xf1,0xf5,0x4d,0x58,0xca]
vaddpd zmm1{k5},zmm1,zmm2

// CHECK: vaddpd  %zmm2, %zmm1, %zmm1 {%k5} {z}
// CHECK:  encoding: [0x62,0xf1,0xf5,0xcd,0x58,0xca]
vaddpd zmm1{k5} {z},zmm1,zmm2

// CHECK: vaddpd  {rn-sae}, %zmm2, %zmm1, %zmm1
// CHECK:  encoding: [0x62,0xf1,0xf5,0x18,0x58,0xca]
vaddpd zmm1,zmm1,zmm2,{rn-sae}

// CHECK: vaddpd  {ru-sae}, %zmm2, %zmm1, %zmm1
// CHECK:  encoding: [0x62,0xf1,0xf5,0x58,0x58,0xca]
vaddpd zmm1,zmm1,zmm2,{ru-sae}

// CHECK:  vaddpd  {rd-sae}, %zmm2, %zmm1, %zmm1
// CHECK:  encoding: [0x62,0xf1,0xf5,0x38,0x58,0xca]
vaddpd zmm1,zmm1,zmm2,{rd-sae}

// CHECK: vaddpd  {rz-sae}, %zmm2, %zmm1, %zmm1
// CHECK:  encoding: [0x62,0xf1,0xf5,0x78,0x58,0xca]
vaddpd zmm1,zmm1,zmm2,{rz-sae}

