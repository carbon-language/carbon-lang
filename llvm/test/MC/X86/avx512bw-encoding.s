// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx  --show-encoding %s | FileCheck %s

// CHECK: vpblendmb %zmm25, %zmm18, %zmm17
// CHECK:  encoding: [0x62,0x82,0x6d,0x40,0x66,0xc9]
          vpblendmb %zmm25, %zmm18, %zmm17

// CHECK: vpblendmb %zmm25, %zmm18, %zmm17 {%k5}
// CHECK:  encoding: [0x62,0x82,0x6d,0x45,0x66,0xc9]
          vpblendmb %zmm25, %zmm18, %zmm17 {%k5}

// CHECK: vpblendmb %zmm25, %zmm18, %zmm17 {%k5} {z}
// CHECK:  encoding: [0x62,0x82,0x6d,0xc5,0x66,0xc9]
          vpblendmb %zmm25, %zmm18, %zmm17 {%k5} {z}

// CHECK: vpblendmb (%rcx), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xe2,0x6d,0x40,0x66,0x09]
          vpblendmb (%rcx), %zmm18, %zmm17

// CHECK: vpblendmb 291(%rax,%r14,8), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xa2,0x6d,0x40,0x66,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vpblendmb 291(%rax,%r14,8), %zmm18, %zmm17

// CHECK: vpblendmb 8128(%rdx), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xe2,0x6d,0x40,0x66,0x4a,0x7f]
          vpblendmb 8128(%rdx), %zmm18, %zmm17

// CHECK: vpblendmb 8192(%rdx), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xe2,0x6d,0x40,0x66,0x8a,0x00,0x20,0x00,0x00]
          vpblendmb 8192(%rdx), %zmm18, %zmm17

// CHECK: vpblendmb -8192(%rdx), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xe2,0x6d,0x40,0x66,0x4a,0x80]
          vpblendmb -8192(%rdx), %zmm18, %zmm17

// CHECK: vpblendmb -8256(%rdx), %zmm18, %zmm17
// CHECK:  encoding: [0x62,0xe2,0x6d,0x40,0x66,0x8a,0xc0,0xdf,0xff,0xff]
          vpblendmb -8256(%rdx), %zmm18, %zmm17

// CHECK: vpblendmw %zmm17, %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x22,0xdd,0x40,0x66,0xd1]
          vpblendmw %zmm17, %zmm20, %zmm26

// CHECK: vpblendmw %zmm17, %zmm20, %zmm26 {%k7}
// CHECK:  encoding: [0x62,0x22,0xdd,0x47,0x66,0xd1]
          vpblendmw %zmm17, %zmm20, %zmm26 {%k7}

// CHECK: vpblendmw %zmm17, %zmm20, %zmm26 {%k7} {z}
// CHECK:  encoding: [0x62,0x22,0xdd,0xc7,0x66,0xd1]
          vpblendmw %zmm17, %zmm20, %zmm26 {%k7} {z}

// CHECK: vpblendmw (%rcx), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x62,0xdd,0x40,0x66,0x11]
          vpblendmw (%rcx), %zmm20, %zmm26

// CHECK: vpblendmw 291(%rax,%r14,8), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x22,0xdd,0x40,0x66,0x94,0xf0,0x23,0x01,0x00,0x00]
          vpblendmw 291(%rax,%r14,8), %zmm20, %zmm26

// CHECK: vpblendmw 8128(%rdx), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x62,0xdd,0x40,0x66,0x52,0x7f]
          vpblendmw 8128(%rdx), %zmm20, %zmm26

// CHECK: vpblendmw 8192(%rdx), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x62,0xdd,0x40,0x66,0x92,0x00,0x20,0x00,0x00]
          vpblendmw 8192(%rdx), %zmm20, %zmm26

// CHECK: vpblendmw -8192(%rdx), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x62,0xdd,0x40,0x66,0x52,0x80]
          vpblendmw -8192(%rdx), %zmm20, %zmm26

// CHECK: vpblendmw -8256(%rdx), %zmm20, %zmm26
// CHECK:  encoding: [0x62,0x62,0xdd,0x40,0x66,0x92,0xc0,0xdf,0xff,0xff]
          vpblendmw -8256(%rdx), %zmm20, %zmm26
