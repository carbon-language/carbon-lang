// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vpclmulqdq  $17, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x11]
          vpclmulhqhqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $17, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x11]
          vpclmulhqhqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $1, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x01]
          vpclmulhqlqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $1, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x01]
          vpclmulhqlqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $16, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x10]
          vpclmullqhqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $16, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x10]
          vpclmullqhqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $0, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x00]
          vpclmullqlqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $0, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x00]
          vpclmullqlqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $17, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x11]
          vpclmulqdq  $17, %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $17, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x11]
          vpclmulqdq  $17, (%eax), %xmm5, %xmm3

