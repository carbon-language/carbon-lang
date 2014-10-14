// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512dq -mattr=+avx512vl  --show-encoding %s | FileCheck %s

// CHECK: vpmullq %xmm22, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x22,0xf5,0x00,0x40,0xd6]
          vpmullq %xmm22, %xmm17, %xmm26

// CHECK: vpmullq %xmm22, %xmm17, %xmm26 {%k6}
// CHECK:  encoding: [0x62,0x22,0xf5,0x06,0x40,0xd6]
          vpmullq %xmm22, %xmm17, %xmm26 {%k6}

// CHECK: vpmullq %xmm22, %xmm17, %xmm26 {%k6} {z}
// CHECK:  encoding: [0x62,0x22,0xf5,0x86,0x40,0xd6]
          vpmullq %xmm22, %xmm17, %xmm26 {%k6} {z}

// CHECK: vpmullq (%rcx), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x00,0x40,0x11]
          vpmullq (%rcx), %xmm17, %xmm26

// CHECK: vpmullq 291(%rax,%r14,8), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x22,0xf5,0x00,0x40,0x94,0xf0,0x23,0x01,0x00,0x00]
          vpmullq 291(%rax,%r14,8), %xmm17, %xmm26

// CHECK: vpmullq (%rcx){1to2}, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x10,0x40,0x11]
          vpmullq (%rcx){1to2}, %xmm17, %xmm26

// CHECK: vpmullq 2032(%rdx), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x00,0x40,0x52,0x7f]
          vpmullq 2032(%rdx), %xmm17, %xmm26

// CHECK: vpmullq 2048(%rdx), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x00,0x40,0x92,0x00,0x08,0x00,0x00]
          vpmullq 2048(%rdx), %xmm17, %xmm26

// CHECK: vpmullq -2048(%rdx), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x00,0x40,0x52,0x80]
          vpmullq -2048(%rdx), %xmm17, %xmm26

// CHECK: vpmullq -2064(%rdx), %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x00,0x40,0x92,0xf0,0xf7,0xff,0xff]
          vpmullq -2064(%rdx), %xmm17, %xmm26

// CHECK: vpmullq 1016(%rdx){1to2}, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x10,0x40,0x52,0x7f]
          vpmullq 1016(%rdx){1to2}, %xmm17, %xmm26

// CHECK: vpmullq 1024(%rdx){1to2}, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x10,0x40,0x92,0x00,0x04,0x00,0x00]
          vpmullq 1024(%rdx){1to2}, %xmm17, %xmm26

// CHECK: vpmullq -1024(%rdx){1to2}, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x10,0x40,0x52,0x80]
          vpmullq -1024(%rdx){1to2}, %xmm17, %xmm26

// CHECK: vpmullq -1032(%rdx){1to2}, %xmm17, %xmm26
// CHECK:  encoding: [0x62,0x62,0xf5,0x10,0x40,0x92,0xf8,0xfb,0xff,0xff]
          vpmullq -1032(%rdx){1to2}, %xmm17, %xmm26

// CHECK: vpmullq %ymm25, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x02,0xb5,0x20,0x40,0xc9]
          vpmullq %ymm25, %ymm25, %ymm25

// CHECK: vpmullq %ymm25, %ymm25, %ymm25 {%k3}
// CHECK:  encoding: [0x62,0x02,0xb5,0x23,0x40,0xc9]
          vpmullq %ymm25, %ymm25, %ymm25 {%k3}

// CHECK: vpmullq %ymm25, %ymm25, %ymm25 {%k3} {z}
// CHECK:  encoding: [0x62,0x02,0xb5,0xa3,0x40,0xc9]
          vpmullq %ymm25, %ymm25, %ymm25 {%k3} {z}

// CHECK: vpmullq (%rcx), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x20,0x40,0x09]
          vpmullq (%rcx), %ymm25, %ymm25

// CHECK: vpmullq 291(%rax,%r14,8), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x22,0xb5,0x20,0x40,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vpmullq 291(%rax,%r14,8), %ymm25, %ymm25

// CHECK: vpmullq (%rcx){1to4}, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x30,0x40,0x09]
          vpmullq (%rcx){1to4}, %ymm25, %ymm25

// CHECK: vpmullq 4064(%rdx), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x20,0x40,0x4a,0x7f]
          vpmullq 4064(%rdx), %ymm25, %ymm25

// CHECK: vpmullq 4096(%rdx), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x20,0x40,0x8a,0x00,0x10,0x00,0x00]
          vpmullq 4096(%rdx), %ymm25, %ymm25

// CHECK: vpmullq -4096(%rdx), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x20,0x40,0x4a,0x80]
          vpmullq -4096(%rdx), %ymm25, %ymm25

// CHECK: vpmullq -4128(%rdx), %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x20,0x40,0x8a,0xe0,0xef,0xff,0xff]
          vpmullq -4128(%rdx), %ymm25, %ymm25

// CHECK: vpmullq 1016(%rdx){1to4}, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x30,0x40,0x4a,0x7f]
          vpmullq 1016(%rdx){1to4}, %ymm25, %ymm25

// CHECK: vpmullq 1024(%rdx){1to4}, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x30,0x40,0x8a,0x00,0x04,0x00,0x00]
          vpmullq 1024(%rdx){1to4}, %ymm25, %ymm25

// CHECK: vpmullq -1024(%rdx){1to4}, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x30,0x40,0x4a,0x80]
          vpmullq -1024(%rdx){1to4}, %ymm25, %ymm25

// CHECK: vpmullq -1032(%rdx){1to4}, %ymm25, %ymm25
// CHECK:  encoding: [0x62,0x62,0xb5,0x30,0x40,0x8a,0xf8,0xfb,0xff,0xff]
          vpmullq -1032(%rdx){1to4}, %ymm25, %ymm25
