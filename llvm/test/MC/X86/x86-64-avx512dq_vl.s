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

// CHECK: vandpd %xmm20, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xa1,0x95,0x00,0x54,0xec]
          vandpd %xmm20, %xmm29, %xmm21

// CHECK: vandpd %xmm20, %xmm29, %xmm21 {%k6}
// CHECK:  encoding: [0x62,0xa1,0x95,0x06,0x54,0xec]
          vandpd %xmm20, %xmm29, %xmm21 {%k6}

// CHECK: vandpd %xmm20, %xmm29, %xmm21 {%k6} {z}
// CHECK:  encoding: [0x62,0xa1,0x95,0x86,0x54,0xec]
          vandpd %xmm20, %xmm29, %xmm21 {%k6} {z}

// CHECK: vandpd (%rcx), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x00,0x54,0x29]
          vandpd (%rcx), %xmm29, %xmm21

// CHECK: vandpd 291(%rax,%r14,8), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xa1,0x95,0x00,0x54,0xac,0xf0,0x23,0x01,0x00,0x00]
          vandpd 291(%rax,%r14,8), %xmm29, %xmm21

// CHECK: vandpd (%rcx){1to2}, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x10,0x54,0x29]
          vandpd (%rcx){1to2}, %xmm29, %xmm21

// CHECK: vandpd 2032(%rdx), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x00,0x54,0x6a,0x7f]
          vandpd 2032(%rdx), %xmm29, %xmm21

// CHECK: vandpd 2048(%rdx), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x00,0x54,0xaa,0x00,0x08,0x00,0x00]
          vandpd 2048(%rdx), %xmm29, %xmm21

// CHECK: vandpd -2048(%rdx), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x00,0x54,0x6a,0x80]
          vandpd -2048(%rdx), %xmm29, %xmm21

// CHECK: vandpd -2064(%rdx), %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x00,0x54,0xaa,0xf0,0xf7,0xff,0xff]
          vandpd -2064(%rdx), %xmm29, %xmm21

// CHECK: vandpd 1016(%rdx){1to2}, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x10,0x54,0x6a,0x7f]
          vandpd 1016(%rdx){1to2}, %xmm29, %xmm21

// CHECK: vandpd 1024(%rdx){1to2}, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x10,0x54,0xaa,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to2}, %xmm29, %xmm21

// CHECK: vandpd -1024(%rdx){1to2}, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x10,0x54,0x6a,0x80]
          vandpd -1024(%rdx){1to2}, %xmm29, %xmm21

// CHECK: vandpd -1032(%rdx){1to2}, %xmm29, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x95,0x10,0x54,0xaa,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to2}, %xmm29, %xmm21

// CHECK: vandpd %ymm28, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x01,0xd5,0x20,0x54,0xe4]
          vandpd %ymm28, %ymm21, %ymm28

// CHECK: vandpd %ymm28, %ymm21, %ymm28 {%k4}
// CHECK:  encoding: [0x62,0x01,0xd5,0x24,0x54,0xe4]
          vandpd %ymm28, %ymm21, %ymm28 {%k4}

// CHECK: vandpd %ymm28, %ymm21, %ymm28 {%k4} {z}
// CHECK:  encoding: [0x62,0x01,0xd5,0xa4,0x54,0xe4]
          vandpd %ymm28, %ymm21, %ymm28 {%k4} {z}

// CHECK: vandpd (%rcx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x54,0x21]
          vandpd (%rcx), %ymm21, %ymm28

// CHECK: vandpd 291(%rax,%r14,8), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x21,0xd5,0x20,0x54,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vandpd 291(%rax,%r14,8), %ymm21, %ymm28

// CHECK: vandpd (%rcx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x54,0x21]
          vandpd (%rcx){1to4}, %ymm21, %ymm28

// CHECK: vandpd 4064(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x54,0x62,0x7f]
          vandpd 4064(%rdx), %ymm21, %ymm28

// CHECK: vandpd 4096(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x54,0xa2,0x00,0x10,0x00,0x00]
          vandpd 4096(%rdx), %ymm21, %ymm28

// CHECK: vandpd -4096(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x54,0x62,0x80]
          vandpd -4096(%rdx), %ymm21, %ymm28

// CHECK: vandpd -4128(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x54,0xa2,0xe0,0xef,0xff,0xff]
          vandpd -4128(%rdx), %ymm21, %ymm28

// CHECK: vandpd 1016(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x54,0x62,0x7f]
          vandpd 1016(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vandpd 1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x54,0xa2,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vandpd -1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x54,0x62,0x80]
          vandpd -1024(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vandpd -1032(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x54,0xa2,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vandps %xmm24, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0x81,0x54,0x00,0x54,0xf8]
          vandps %xmm24, %xmm21, %xmm23

// CHECK: vandps %xmm24, %xmm21, %xmm23 {%k5}
// CHECK:  encoding: [0x62,0x81,0x54,0x05,0x54,0xf8]
          vandps %xmm24, %xmm21, %xmm23 {%k5}

// CHECK: vandps %xmm24, %xmm21, %xmm23 {%k5} {z}
// CHECK:  encoding: [0x62,0x81,0x54,0x85,0x54,0xf8]
          vandps %xmm24, %xmm21, %xmm23 {%k5} {z}

// CHECK: vandps (%rcx), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x54,0x39]
          vandps (%rcx), %xmm21, %xmm23

// CHECK: vandps 291(%rax,%r14,8), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xa1,0x54,0x00,0x54,0xbc,0xf0,0x23,0x01,0x00,0x00]
          vandps 291(%rax,%r14,8), %xmm21, %xmm23

// CHECK: vandps (%rcx){1to4}, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x54,0x39]
          vandps (%rcx){1to4}, %xmm21, %xmm23

// CHECK: vandps 2032(%rdx), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x54,0x7a,0x7f]
          vandps 2032(%rdx), %xmm21, %xmm23

// CHECK: vandps 2048(%rdx), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x54,0xba,0x00,0x08,0x00,0x00]
          vandps 2048(%rdx), %xmm21, %xmm23

// CHECK: vandps -2048(%rdx), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x54,0x7a,0x80]
          vandps -2048(%rdx), %xmm21, %xmm23

// CHECK: vandps -2064(%rdx), %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x54,0xba,0xf0,0xf7,0xff,0xff]
          vandps -2064(%rdx), %xmm21, %xmm23

// CHECK: vandps 508(%rdx){1to4}, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x54,0x7a,0x7f]
          vandps 508(%rdx){1to4}, %xmm21, %xmm23

// CHECK: vandps 512(%rdx){1to4}, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x54,0xba,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to4}, %xmm21, %xmm23

// CHECK: vandps -512(%rdx){1to4}, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x54,0x7a,0x80]
          vandps -512(%rdx){1to4}, %xmm21, %xmm23

// CHECK: vandps -516(%rdx){1to4}, %xmm21, %xmm23
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x54,0xba,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to4}, %xmm21, %xmm23

// CHECK: vandps %ymm23, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x21,0x6c,0x20,0x54,0xd7]
          vandps %ymm23, %ymm18, %ymm26

// CHECK: vandps %ymm23, %ymm18, %ymm26 {%k6}
// CHECK:  encoding: [0x62,0x21,0x6c,0x26,0x54,0xd7]
          vandps %ymm23, %ymm18, %ymm26 {%k6}

// CHECK: vandps %ymm23, %ymm18, %ymm26 {%k6} {z}
// CHECK:  encoding: [0x62,0x21,0x6c,0xa6,0x54,0xd7]
          vandps %ymm23, %ymm18, %ymm26 {%k6} {z}

// CHECK: vandps (%rcx), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x20,0x54,0x11]
          vandps (%rcx), %ymm18, %ymm26

// CHECK: vandps 291(%rax,%r14,8), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x21,0x6c,0x20,0x54,0x94,0xf0,0x23,0x01,0x00,0x00]
          vandps 291(%rax,%r14,8), %ymm18, %ymm26

// CHECK: vandps (%rcx){1to8}, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x30,0x54,0x11]
          vandps (%rcx){1to8}, %ymm18, %ymm26

// CHECK: vandps 4064(%rdx), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x20,0x54,0x52,0x7f]
          vandps 4064(%rdx), %ymm18, %ymm26

// CHECK: vandps 4096(%rdx), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x20,0x54,0x92,0x00,0x10,0x00,0x00]
          vandps 4096(%rdx), %ymm18, %ymm26

// CHECK: vandps -4096(%rdx), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x20,0x54,0x52,0x80]
          vandps -4096(%rdx), %ymm18, %ymm26

// CHECK: vandps -4128(%rdx), %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x20,0x54,0x92,0xe0,0xef,0xff,0xff]
          vandps -4128(%rdx), %ymm18, %ymm26

// CHECK: vandps 508(%rdx){1to8}, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x30,0x54,0x52,0x7f]
          vandps 508(%rdx){1to8}, %ymm18, %ymm26

// CHECK: vandps 512(%rdx){1to8}, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x30,0x54,0x92,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to8}, %ymm18, %ymm26

// CHECK: vandps -512(%rdx){1to8}, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x30,0x54,0x52,0x80]
          vandps -512(%rdx){1to8}, %ymm18, %ymm26

// CHECK: vandps -516(%rdx){1to8}, %ymm18, %ymm26
// CHECK:  encoding: [0x62,0x61,0x6c,0x30,0x54,0x92,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to8}, %ymm18, %ymm26

// CHECK: vandnpd %xmm25, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x01,0xa5,0x00,0x55,0xc9]
          vandnpd %xmm25, %xmm27, %xmm25

// CHECK: vandnpd %xmm25, %xmm27, %xmm25 {%k5}
// CHECK:  encoding: [0x62,0x01,0xa5,0x05,0x55,0xc9]
          vandnpd %xmm25, %xmm27, %xmm25 {%k5}

// CHECK: vandnpd %xmm25, %xmm27, %xmm25 {%k5} {z}
// CHECK:  encoding: [0x62,0x01,0xa5,0x85,0x55,0xc9]
          vandnpd %xmm25, %xmm27, %xmm25 {%k5} {z}

// CHECK: vandnpd (%rcx), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x00,0x55,0x09]
          vandnpd (%rcx), %xmm27, %xmm25

// CHECK: vandnpd 291(%rax,%r14,8), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x21,0xa5,0x00,0x55,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vandnpd 291(%rax,%r14,8), %xmm27, %xmm25

// CHECK: vandnpd (%rcx){1to2}, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x10,0x55,0x09]
          vandnpd (%rcx){1to2}, %xmm27, %xmm25

// CHECK: vandnpd 2032(%rdx), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x00,0x55,0x4a,0x7f]
          vandnpd 2032(%rdx), %xmm27, %xmm25

// CHECK: vandnpd 2048(%rdx), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x00,0x55,0x8a,0x00,0x08,0x00,0x00]
          vandnpd 2048(%rdx), %xmm27, %xmm25

// CHECK: vandnpd -2048(%rdx), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x00,0x55,0x4a,0x80]
          vandnpd -2048(%rdx), %xmm27, %xmm25

// CHECK: vandnpd -2064(%rdx), %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x00,0x55,0x8a,0xf0,0xf7,0xff,0xff]
          vandnpd -2064(%rdx), %xmm27, %xmm25

// CHECK: vandnpd 1016(%rdx){1to2}, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x10,0x55,0x4a,0x7f]
          vandnpd 1016(%rdx){1to2}, %xmm27, %xmm25

// CHECK: vandnpd 1024(%rdx){1to2}, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x10,0x55,0x8a,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to2}, %xmm27, %xmm25

// CHECK: vandnpd -1024(%rdx){1to2}, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x10,0x55,0x4a,0x80]
          vandnpd -1024(%rdx){1to2}, %xmm27, %xmm25

// CHECK: vandnpd -1032(%rdx){1to2}, %xmm27, %xmm25
// CHECK:  encoding: [0x62,0x61,0xa5,0x10,0x55,0x8a,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to2}, %xmm27, %xmm25

// CHECK: vandnpd %ymm22, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xa1,0xed,0x20,0x55,0xf6]
          vandnpd %ymm22, %ymm18, %ymm22

// CHECK: vandnpd %ymm22, %ymm18, %ymm22 {%k7}
// CHECK:  encoding: [0x62,0xa1,0xed,0x27,0x55,0xf6]
          vandnpd %ymm22, %ymm18, %ymm22 {%k7}

// CHECK: vandnpd %ymm22, %ymm18, %ymm22 {%k7} {z}
// CHECK:  encoding: [0x62,0xa1,0xed,0xa7,0x55,0xf6]
          vandnpd %ymm22, %ymm18, %ymm22 {%k7} {z}

// CHECK: vandnpd (%rcx), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x20,0x55,0x31]
          vandnpd (%rcx), %ymm18, %ymm22

// CHECK: vandnpd 291(%rax,%r14,8), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xa1,0xed,0x20,0x55,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vandnpd 291(%rax,%r14,8), %ymm18, %ymm22

// CHECK: vandnpd (%rcx){1to4}, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x30,0x55,0x31]
          vandnpd (%rcx){1to4}, %ymm18, %ymm22

// CHECK: vandnpd 4064(%rdx), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x20,0x55,0x72,0x7f]
          vandnpd 4064(%rdx), %ymm18, %ymm22

// CHECK: vandnpd 4096(%rdx), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x20,0x55,0xb2,0x00,0x10,0x00,0x00]
          vandnpd 4096(%rdx), %ymm18, %ymm22

// CHECK: vandnpd -4096(%rdx), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x20,0x55,0x72,0x80]
          vandnpd -4096(%rdx), %ymm18, %ymm22

// CHECK: vandnpd -4128(%rdx), %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x20,0x55,0xb2,0xe0,0xef,0xff,0xff]
          vandnpd -4128(%rdx), %ymm18, %ymm22

// CHECK: vandnpd 1016(%rdx){1to4}, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x30,0x55,0x72,0x7f]
          vandnpd 1016(%rdx){1to4}, %ymm18, %ymm22

// CHECK: vandnpd 1024(%rdx){1to4}, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x30,0x55,0xb2,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to4}, %ymm18, %ymm22

// CHECK: vandnpd -1024(%rdx){1to4}, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x30,0x55,0x72,0x80]
          vandnpd -1024(%rdx){1to4}, %ymm18, %ymm22

// CHECK: vandnpd -1032(%rdx){1to4}, %ymm18, %ymm22
// CHECK:  encoding: [0x62,0xe1,0xed,0x30,0x55,0xb2,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to4}, %ymm18, %ymm22

// CHECK: vandnps %xmm27, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0x81,0x54,0x00,0x55,0xeb]
          vandnps %xmm27, %xmm21, %xmm21

// CHECK: vandnps %xmm27, %xmm21, %xmm21 {%k2}
// CHECK:  encoding: [0x62,0x81,0x54,0x02,0x55,0xeb]
          vandnps %xmm27, %xmm21, %xmm21 {%k2}

// CHECK: vandnps %xmm27, %xmm21, %xmm21 {%k2} {z}
// CHECK:  encoding: [0x62,0x81,0x54,0x82,0x55,0xeb]
          vandnps %xmm27, %xmm21, %xmm21 {%k2} {z}

// CHECK: vandnps (%rcx), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x55,0x29]
          vandnps (%rcx), %xmm21, %xmm21

// CHECK: vandnps 291(%rax,%r14,8), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xa1,0x54,0x00,0x55,0xac,0xf0,0x23,0x01,0x00,0x00]
          vandnps 291(%rax,%r14,8), %xmm21, %xmm21

// CHECK: vandnps (%rcx){1to4}, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x55,0x29]
          vandnps (%rcx){1to4}, %xmm21, %xmm21

// CHECK: vandnps 2032(%rdx), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x55,0x6a,0x7f]
          vandnps 2032(%rdx), %xmm21, %xmm21

// CHECK: vandnps 2048(%rdx), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x55,0xaa,0x00,0x08,0x00,0x00]
          vandnps 2048(%rdx), %xmm21, %xmm21

// CHECK: vandnps -2048(%rdx), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x55,0x6a,0x80]
          vandnps -2048(%rdx), %xmm21, %xmm21

// CHECK: vandnps -2064(%rdx), %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x55,0xaa,0xf0,0xf7,0xff,0xff]
          vandnps -2064(%rdx), %xmm21, %xmm21

// CHECK: vandnps 508(%rdx){1to4}, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x55,0x6a,0x7f]
          vandnps 508(%rdx){1to4}, %xmm21, %xmm21

// CHECK: vandnps 512(%rdx){1to4}, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x55,0xaa,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to4}, %xmm21, %xmm21

// CHECK: vandnps -512(%rdx){1to4}, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x55,0x6a,0x80]
          vandnps -512(%rdx){1to4}, %xmm21, %xmm21

// CHECK: vandnps -516(%rdx){1to4}, %xmm21, %xmm21
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x55,0xaa,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to4}, %xmm21, %xmm21

// CHECK: vandnps %ymm25, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0x81,0x44,0x20,0x55,0xd9]
          vandnps %ymm25, %ymm23, %ymm19

// CHECK: vandnps %ymm25, %ymm23, %ymm19 {%k1}
// CHECK:  encoding: [0x62,0x81,0x44,0x21,0x55,0xd9]
          vandnps %ymm25, %ymm23, %ymm19 {%k1}

// CHECK: vandnps %ymm25, %ymm23, %ymm19 {%k1} {z}
// CHECK:  encoding: [0x62,0x81,0x44,0xa1,0x55,0xd9]
          vandnps %ymm25, %ymm23, %ymm19 {%k1} {z}

// CHECK: vandnps (%rcx), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x20,0x55,0x19]
          vandnps (%rcx), %ymm23, %ymm19

// CHECK: vandnps 291(%rax,%r14,8), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xa1,0x44,0x20,0x55,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vandnps 291(%rax,%r14,8), %ymm23, %ymm19

// CHECK: vandnps (%rcx){1to8}, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x30,0x55,0x19]
          vandnps (%rcx){1to8}, %ymm23, %ymm19

// CHECK: vandnps 4064(%rdx), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x20,0x55,0x5a,0x7f]
          vandnps 4064(%rdx), %ymm23, %ymm19

// CHECK: vandnps 4096(%rdx), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x20,0x55,0x9a,0x00,0x10,0x00,0x00]
          vandnps 4096(%rdx), %ymm23, %ymm19

// CHECK: vandnps -4096(%rdx), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x20,0x55,0x5a,0x80]
          vandnps -4096(%rdx), %ymm23, %ymm19

// CHECK: vandnps -4128(%rdx), %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x20,0x55,0x9a,0xe0,0xef,0xff,0xff]
          vandnps -4128(%rdx), %ymm23, %ymm19

// CHECK: vandnps 508(%rdx){1to8}, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x30,0x55,0x5a,0x7f]
          vandnps 508(%rdx){1to8}, %ymm23, %ymm19

// CHECK: vandnps 512(%rdx){1to8}, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x30,0x55,0x9a,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to8}, %ymm23, %ymm19

// CHECK: vandnps -512(%rdx){1to8}, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x30,0x55,0x5a,0x80]
          vandnps -512(%rdx){1to8}, %ymm23, %ymm19

// CHECK: vandnps -516(%rdx){1to8}, %ymm23, %ymm19
// CHECK:  encoding: [0x62,0xe1,0x44,0x30,0x55,0x9a,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to8}, %ymm23, %ymm19

// CHECK: vorpd  %xmm18, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xa1,0xa5,0x00,0x56,0xfa]
          vorpd  %xmm18, %xmm27, %xmm23

// CHECK: vorpd  %xmm18, %xmm27, %xmm23 {%k1}
// CHECK:  encoding: [0x62,0xa1,0xa5,0x01,0x56,0xfa]
          vorpd  %xmm18, %xmm27, %xmm23 {%k1}

// CHECK: vorpd  %xmm18, %xmm27, %xmm23 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0xa5,0x81,0x56,0xfa]
          vorpd  %xmm18, %xmm27, %xmm23 {%k1} {z}

// CHECK: vorpd  (%rcx), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x00,0x56,0x39]
          vorpd  (%rcx), %xmm27, %xmm23

// CHECK: vorpd  291(%rax,%r14,8), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xa1,0xa5,0x00,0x56,0xbc,0xf0,0x23,0x01,0x00,0x00]
          vorpd  291(%rax,%r14,8), %xmm27, %xmm23

// CHECK: vorpd  (%rcx){1to2}, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x10,0x56,0x39]
          vorpd  (%rcx){1to2}, %xmm27, %xmm23

// CHECK: vorpd  2032(%rdx), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x00,0x56,0x7a,0x7f]
          vorpd  2032(%rdx), %xmm27, %xmm23

// CHECK: vorpd  2048(%rdx), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x00,0x56,0xba,0x00,0x08,0x00,0x00]
          vorpd  2048(%rdx), %xmm27, %xmm23

// CHECK: vorpd  -2048(%rdx), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x00,0x56,0x7a,0x80]
          vorpd  -2048(%rdx), %xmm27, %xmm23

// CHECK: vorpd  -2064(%rdx), %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x00,0x56,0xba,0xf0,0xf7,0xff,0xff]
          vorpd  -2064(%rdx), %xmm27, %xmm23

// CHECK: vorpd  1016(%rdx){1to2}, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x10,0x56,0x7a,0x7f]
          vorpd  1016(%rdx){1to2}, %xmm27, %xmm23

// CHECK: vorpd  1024(%rdx){1to2}, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x10,0x56,0xba,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to2}, %xmm27, %xmm23

// CHECK: vorpd  -1024(%rdx){1to2}, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x10,0x56,0x7a,0x80]
          vorpd  -1024(%rdx){1to2}, %xmm27, %xmm23

// CHECK: vorpd  -1032(%rdx){1to2}, %xmm27, %xmm23
// CHECK:  encoding: [0x62,0xe1,0xa5,0x10,0x56,0xba,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to2}, %xmm27, %xmm23

// CHECK: vorpd  %ymm20, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x56,0xd4]
          vorpd  %ymm20, %ymm29, %ymm26

// CHECK: vorpd  %ymm20, %ymm29, %ymm26 {%k5}
// CHECK:  encoding: [0x62,0x21,0x95,0x25,0x56,0xd4]
          vorpd  %ymm20, %ymm29, %ymm26 {%k5}

// CHECK: vorpd  %ymm20, %ymm29, %ymm26 {%k5} {z}
// CHECK:  encoding: [0x62,0x21,0x95,0xa5,0x56,0xd4]
          vorpd  %ymm20, %ymm29, %ymm26 {%k5} {z}

// CHECK: vorpd  (%rcx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x56,0x11]
          vorpd  (%rcx), %ymm29, %ymm26

// CHECK: vorpd  291(%rax,%r14,8), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x56,0x94,0xf0,0x23,0x01,0x00,0x00]
          vorpd  291(%rax,%r14,8), %ymm29, %ymm26

// CHECK: vorpd  (%rcx){1to4}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x56,0x11]
          vorpd  (%rcx){1to4}, %ymm29, %ymm26

// CHECK: vorpd  4064(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x56,0x52,0x7f]
          vorpd  4064(%rdx), %ymm29, %ymm26

// CHECK: vorpd  4096(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x56,0x92,0x00,0x10,0x00,0x00]
          vorpd  4096(%rdx), %ymm29, %ymm26

// CHECK: vorpd  -4096(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x56,0x52,0x80]
          vorpd  -4096(%rdx), %ymm29, %ymm26

// CHECK: vorpd  -4128(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x56,0x92,0xe0,0xef,0xff,0xff]
          vorpd  -4128(%rdx), %ymm29, %ymm26

// CHECK: vorpd  1016(%rdx){1to4}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x56,0x52,0x7f]
          vorpd  1016(%rdx){1to4}, %ymm29, %ymm26

// CHECK: vorpd  1024(%rdx){1to4}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x56,0x92,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to4}, %ymm29, %ymm26

// CHECK: vorpd  -1024(%rdx){1to4}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x56,0x52,0x80]
          vorpd  -1024(%rdx){1to4}, %ymm29, %ymm26

// CHECK: vorpd  -1032(%rdx){1to4}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x56,0x92,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to4}, %ymm29, %ymm26

// CHECK: vorps  %xmm27, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0x81,0x1c,0x00,0x56,0xdb]
          vorps  %xmm27, %xmm28, %xmm19

// CHECK: vorps  %xmm27, %xmm28, %xmm19 {%k4}
// CHECK:  encoding: [0x62,0x81,0x1c,0x04,0x56,0xdb]
          vorps  %xmm27, %xmm28, %xmm19 {%k4}

// CHECK: vorps  %xmm27, %xmm28, %xmm19 {%k4} {z}
// CHECK:  encoding: [0x62,0x81,0x1c,0x84,0x56,0xdb]
          vorps  %xmm27, %xmm28, %xmm19 {%k4} {z}

// CHECK: vorps  (%rcx), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x00,0x56,0x19]
          vorps  (%rcx), %xmm28, %xmm19

// CHECK: vorps  291(%rax,%r14,8), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xa1,0x1c,0x00,0x56,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vorps  291(%rax,%r14,8), %xmm28, %xmm19

// CHECK: vorps  (%rcx){1to4}, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x10,0x56,0x19]
          vorps  (%rcx){1to4}, %xmm28, %xmm19

// CHECK: vorps  2032(%rdx), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x00,0x56,0x5a,0x7f]
          vorps  2032(%rdx), %xmm28, %xmm19

// CHECK: vorps  2048(%rdx), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x00,0x56,0x9a,0x00,0x08,0x00,0x00]
          vorps  2048(%rdx), %xmm28, %xmm19

// CHECK: vorps  -2048(%rdx), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x00,0x56,0x5a,0x80]
          vorps  -2048(%rdx), %xmm28, %xmm19

// CHECK: vorps  -2064(%rdx), %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x00,0x56,0x9a,0xf0,0xf7,0xff,0xff]
          vorps  -2064(%rdx), %xmm28, %xmm19

// CHECK: vorps  508(%rdx){1to4}, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x10,0x56,0x5a,0x7f]
          vorps  508(%rdx){1to4}, %xmm28, %xmm19

// CHECK: vorps  512(%rdx){1to4}, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x10,0x56,0x9a,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to4}, %xmm28, %xmm19

// CHECK: vorps  -512(%rdx){1to4}, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x10,0x56,0x5a,0x80]
          vorps  -512(%rdx){1to4}, %xmm28, %xmm19

// CHECK: vorps  -516(%rdx){1to4}, %xmm28, %xmm19
// CHECK:  encoding: [0x62,0xe1,0x1c,0x10,0x56,0x9a,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to4}, %xmm28, %xmm19

// CHECK: vorps  %ymm26, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x01,0x2c,0x20,0x56,0xda]
          vorps  %ymm26, %ymm26, %ymm27

// CHECK: vorps  %ymm26, %ymm26, %ymm27 {%k1}
// CHECK:  encoding: [0x62,0x01,0x2c,0x21,0x56,0xda]
          vorps  %ymm26, %ymm26, %ymm27 {%k1}

// CHECK: vorps  %ymm26, %ymm26, %ymm27 {%k1} {z}
// CHECK:  encoding: [0x62,0x01,0x2c,0xa1,0x56,0xda]
          vorps  %ymm26, %ymm26, %ymm27 {%k1} {z}

// CHECK: vorps  (%rcx), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x20,0x56,0x19]
          vorps  (%rcx), %ymm26, %ymm27

// CHECK: vorps  291(%rax,%r14,8), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x21,0x2c,0x20,0x56,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vorps  291(%rax,%r14,8), %ymm26, %ymm27

// CHECK: vorps  (%rcx){1to8}, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x30,0x56,0x19]
          vorps  (%rcx){1to8}, %ymm26, %ymm27

// CHECK: vorps  4064(%rdx), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x20,0x56,0x5a,0x7f]
          vorps  4064(%rdx), %ymm26, %ymm27

// CHECK: vorps  4096(%rdx), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x20,0x56,0x9a,0x00,0x10,0x00,0x00]
          vorps  4096(%rdx), %ymm26, %ymm27

// CHECK: vorps  -4096(%rdx), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x20,0x56,0x5a,0x80]
          vorps  -4096(%rdx), %ymm26, %ymm27

// CHECK: vorps  -4128(%rdx), %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x20,0x56,0x9a,0xe0,0xef,0xff,0xff]
          vorps  -4128(%rdx), %ymm26, %ymm27

// CHECK: vorps  508(%rdx){1to8}, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x30,0x56,0x5a,0x7f]
          vorps  508(%rdx){1to8}, %ymm26, %ymm27

// CHECK: vorps  512(%rdx){1to8}, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x30,0x56,0x9a,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to8}, %ymm26, %ymm27

// CHECK: vorps  -512(%rdx){1to8}, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x30,0x56,0x5a,0x80]
          vorps  -512(%rdx){1to8}, %ymm26, %ymm27

// CHECK: vorps  -516(%rdx){1to8}, %ymm26, %ymm27
// CHECK:  encoding: [0x62,0x61,0x2c,0x30,0x56,0x9a,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to8}, %ymm26, %ymm27

// CHECK: vxorpd %xmm23, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xa1,0xd5,0x00,0x57,0xd7]
          vxorpd %xmm23, %xmm21, %xmm18

// CHECK: vxorpd %xmm23, %xmm21, %xmm18 {%k2}
// CHECK:  encoding: [0x62,0xa1,0xd5,0x02,0x57,0xd7]
          vxorpd %xmm23, %xmm21, %xmm18 {%k2}

// CHECK: vxorpd %xmm23, %xmm21, %xmm18 {%k2} {z}
// CHECK:  encoding: [0x62,0xa1,0xd5,0x82,0x57,0xd7]
          vxorpd %xmm23, %xmm21, %xmm18 {%k2} {z}

// CHECK: vxorpd (%rcx), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x11]
          vxorpd (%rcx), %xmm21, %xmm18

// CHECK: vxorpd 291(%rax,%r14,8), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xa1,0xd5,0x00,0x57,0x94,0xf0,0x23,0x01,0x00,0x00]
          vxorpd 291(%rax,%r14,8), %xmm21, %xmm18

// CHECK: vxorpd (%rcx){1to2}, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x11]
          vxorpd (%rcx){1to2}, %xmm21, %xmm18

// CHECK: vxorpd 2032(%rdx), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x52,0x7f]
          vxorpd 2032(%rdx), %xmm21, %xmm18

// CHECK: vxorpd 2048(%rdx), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x92,0x00,0x08,0x00,0x00]
          vxorpd 2048(%rdx), %xmm21, %xmm18

// CHECK: vxorpd -2048(%rdx), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x52,0x80]
          vxorpd -2048(%rdx), %xmm21, %xmm18

// CHECK: vxorpd -2064(%rdx), %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x92,0xf0,0xf7,0xff,0xff]
          vxorpd -2064(%rdx), %xmm21, %xmm18

// CHECK: vxorpd 1016(%rdx){1to2}, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x52,0x7f]
          vxorpd 1016(%rdx){1to2}, %xmm21, %xmm18

// CHECK: vxorpd 1024(%rdx){1to2}, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x92,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to2}, %xmm21, %xmm18

// CHECK: vxorpd -1024(%rdx){1to2}, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x52,0x80]
          vxorpd -1024(%rdx){1to2}, %xmm21, %xmm18

// CHECK: vxorpd -1032(%rdx){1to2}, %xmm21, %xmm18
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x92,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to2}, %xmm21, %xmm18

// CHECK: vxorpd %ymm19, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x57,0xc3]
          vxorpd %ymm19, %ymm29, %ymm24

// CHECK: vxorpd %ymm19, %ymm29, %ymm24 {%k7}
// CHECK:  encoding: [0x62,0x21,0x95,0x27,0x57,0xc3]
          vxorpd %ymm19, %ymm29, %ymm24 {%k7}

// CHECK: vxorpd %ymm19, %ymm29, %ymm24 {%k7} {z}
// CHECK:  encoding: [0x62,0x21,0x95,0xa7,0x57,0xc3]
          vxorpd %ymm19, %ymm29, %ymm24 {%k7} {z}

// CHECK: vxorpd (%rcx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x57,0x01]
          vxorpd (%rcx), %ymm29, %ymm24

// CHECK: vxorpd 291(%rax,%r14,8), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x57,0x84,0xf0,0x23,0x01,0x00,0x00]
          vxorpd 291(%rax,%r14,8), %ymm29, %ymm24

// CHECK: vxorpd (%rcx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x57,0x01]
          vxorpd (%rcx){1to4}, %ymm29, %ymm24

// CHECK: vxorpd 4064(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x57,0x42,0x7f]
          vxorpd 4064(%rdx), %ymm29, %ymm24

// CHECK: vxorpd 4096(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x57,0x82,0x00,0x10,0x00,0x00]
          vxorpd 4096(%rdx), %ymm29, %ymm24

// CHECK: vxorpd -4096(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x57,0x42,0x80]
          vxorpd -4096(%rdx), %ymm29, %ymm24

// CHECK: vxorpd -4128(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x57,0x82,0xe0,0xef,0xff,0xff]
          vxorpd -4128(%rdx), %ymm29, %ymm24

// CHECK: vxorpd 1016(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x57,0x42,0x7f]
          vxorpd 1016(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vxorpd 1024(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x57,0x82,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vxorpd -1024(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x57,0x42,0x80]
          vxorpd -1024(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vxorpd -1032(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x57,0x82,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vxorps %xmm19, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xa1,0x6c,0x00,0x57,0xe3]
          vxorps %xmm19, %xmm18, %xmm20

// CHECK: vxorps %xmm19, %xmm18, %xmm20 {%k1}
// CHECK:  encoding: [0x62,0xa1,0x6c,0x01,0x57,0xe3]
          vxorps %xmm19, %xmm18, %xmm20 {%k1}

// CHECK: vxorps %xmm19, %xmm18, %xmm20 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0x6c,0x81,0x57,0xe3]
          vxorps %xmm19, %xmm18, %xmm20 {%k1} {z}

// CHECK: vxorps (%rcx), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x00,0x57,0x21]
          vxorps (%rcx), %xmm18, %xmm20

// CHECK: vxorps 291(%rax,%r14,8), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xa1,0x6c,0x00,0x57,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vxorps 291(%rax,%r14,8), %xmm18, %xmm20

// CHECK: vxorps (%rcx){1to4}, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x10,0x57,0x21]
          vxorps (%rcx){1to4}, %xmm18, %xmm20

// CHECK: vxorps 2032(%rdx), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x00,0x57,0x62,0x7f]
          vxorps 2032(%rdx), %xmm18, %xmm20

// CHECK: vxorps 2048(%rdx), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x00,0x57,0xa2,0x00,0x08,0x00,0x00]
          vxorps 2048(%rdx), %xmm18, %xmm20

// CHECK: vxorps -2048(%rdx), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x00,0x57,0x62,0x80]
          vxorps -2048(%rdx), %xmm18, %xmm20

// CHECK: vxorps -2064(%rdx), %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x00,0x57,0xa2,0xf0,0xf7,0xff,0xff]
          vxorps -2064(%rdx), %xmm18, %xmm20

// CHECK: vxorps 508(%rdx){1to4}, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x10,0x57,0x62,0x7f]
          vxorps 508(%rdx){1to4}, %xmm18, %xmm20

// CHECK: vxorps 512(%rdx){1to4}, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x10,0x57,0xa2,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to4}, %xmm18, %xmm20

// CHECK: vxorps -512(%rdx){1to4}, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x10,0x57,0x62,0x80]
          vxorps -512(%rdx){1to4}, %xmm18, %xmm20

// CHECK: vxorps -516(%rdx){1to4}, %xmm18, %xmm20
// CHECK:  encoding: [0x62,0xe1,0x6c,0x10,0x57,0xa2,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to4}, %xmm18, %xmm20

// CHECK: vxorps %ymm24, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x01,0x5c,0x20,0x57,0xd8]
          vxorps %ymm24, %ymm20, %ymm27

// CHECK: vxorps %ymm24, %ymm20, %ymm27 {%k2}
// CHECK:  encoding: [0x62,0x01,0x5c,0x22,0x57,0xd8]
          vxorps %ymm24, %ymm20, %ymm27 {%k2}

// CHECK: vxorps %ymm24, %ymm20, %ymm27 {%k2} {z}
// CHECK:  encoding: [0x62,0x01,0x5c,0xa2,0x57,0xd8]
          vxorps %ymm24, %ymm20, %ymm27 {%k2} {z}

// CHECK: vxorps (%rcx), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x20,0x57,0x19]
          vxorps (%rcx), %ymm20, %ymm27

// CHECK: vxorps 291(%rax,%r14,8), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x21,0x5c,0x20,0x57,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vxorps 291(%rax,%r14,8), %ymm20, %ymm27

// CHECK: vxorps (%rcx){1to8}, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x30,0x57,0x19]
          vxorps (%rcx){1to8}, %ymm20, %ymm27

// CHECK: vxorps 4064(%rdx), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x20,0x57,0x5a,0x7f]
          vxorps 4064(%rdx), %ymm20, %ymm27

// CHECK: vxorps 4096(%rdx), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x20,0x57,0x9a,0x00,0x10,0x00,0x00]
          vxorps 4096(%rdx), %ymm20, %ymm27

// CHECK: vxorps -4096(%rdx), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x20,0x57,0x5a,0x80]
          vxorps -4096(%rdx), %ymm20, %ymm27

// CHECK: vxorps -4128(%rdx), %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x20,0x57,0x9a,0xe0,0xef,0xff,0xff]
          vxorps -4128(%rdx), %ymm20, %ymm27

// CHECK: vxorps 508(%rdx){1to8}, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x30,0x57,0x5a,0x7f]
          vxorps 508(%rdx){1to8}, %ymm20, %ymm27

// CHECK: vxorps 512(%rdx){1to8}, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x30,0x57,0x9a,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to8}, %ymm20, %ymm27

// CHECK: vxorps -512(%rdx){1to8}, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x30,0x57,0x5a,0x80]
          vxorps -512(%rdx){1to8}, %ymm20, %ymm27

// CHECK: vxorps -516(%rdx){1to8}, %ymm20, %ymm27
// CHECK:  encoding: [0x62,0x61,0x5c,0x30,0x57,0x9a,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to8}, %ymm20, %ymm27

// CHECK: vandpd %xmm27, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0x81,0xb5,0x00,0x54,0xdb]
          vandpd %xmm27, %xmm25, %xmm19

// CHECK: vandpd %xmm27, %xmm25, %xmm19 {%k6}
// CHECK:  encoding: [0x62,0x81,0xb5,0x06,0x54,0xdb]
          vandpd %xmm27, %xmm25, %xmm19 {%k6}

// CHECK: vandpd %xmm27, %xmm25, %xmm19 {%k6} {z}
// CHECK:  encoding: [0x62,0x81,0xb5,0x86,0x54,0xdb]
          vandpd %xmm27, %xmm25, %xmm19 {%k6} {z}

// CHECK: vandpd (%rcx), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x00,0x54,0x19]
          vandpd (%rcx), %xmm25, %xmm19

// CHECK: vandpd 4660(%rax,%r14,8), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xa1,0xb5,0x00,0x54,0x9c,0xf0,0x34,0x12,0x00,0x00]
          vandpd 4660(%rax,%r14,8), %xmm25, %xmm19

// CHECK: vandpd (%rcx){1to2}, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x10,0x54,0x19]
          vandpd (%rcx){1to2}, %xmm25, %xmm19

// CHECK: vandpd 2032(%rdx), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x00,0x54,0x5a,0x7f]
          vandpd 2032(%rdx), %xmm25, %xmm19

// CHECK: vandpd 2048(%rdx), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x00,0x54,0x9a,0x00,0x08,0x00,0x00]
          vandpd 2048(%rdx), %xmm25, %xmm19

// CHECK: vandpd -2048(%rdx), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x00,0x54,0x5a,0x80]
          vandpd -2048(%rdx), %xmm25, %xmm19

// CHECK: vandpd -2064(%rdx), %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x00,0x54,0x9a,0xf0,0xf7,0xff,0xff]
          vandpd -2064(%rdx), %xmm25, %xmm19

// CHECK: vandpd 1016(%rdx){1to2}, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x10,0x54,0x5a,0x7f]
          vandpd 1016(%rdx){1to2}, %xmm25, %xmm19

// CHECK: vandpd 1024(%rdx){1to2}, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x10,0x54,0x9a,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to2}, %xmm25, %xmm19

// CHECK: vandpd -1024(%rdx){1to2}, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x10,0x54,0x5a,0x80]
          vandpd -1024(%rdx){1to2}, %xmm25, %xmm19

// CHECK: vandpd -1032(%rdx){1to2}, %xmm25, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xb5,0x10,0x54,0x9a,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to2}, %xmm25, %xmm19

// CHECK: vandpd %ymm21, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x54,0xc5]
          vandpd %ymm21, %ymm29, %ymm24

// CHECK: vandpd %ymm21, %ymm29, %ymm24 {%k2}
// CHECK:  encoding: [0x62,0x21,0x95,0x22,0x54,0xc5]
          vandpd %ymm21, %ymm29, %ymm24 {%k2}

// CHECK: vandpd %ymm21, %ymm29, %ymm24 {%k2} {z}
// CHECK:  encoding: [0x62,0x21,0x95,0xa2,0x54,0xc5]
          vandpd %ymm21, %ymm29, %ymm24 {%k2} {z}

// CHECK: vandpd (%rcx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x54,0x01]
          vandpd (%rcx), %ymm29, %ymm24

// CHECK: vandpd 4660(%rax,%r14,8), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x54,0x84,0xf0,0x34,0x12,0x00,0x00]
          vandpd 4660(%rax,%r14,8), %ymm29, %ymm24

// CHECK: vandpd (%rcx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x54,0x01]
          vandpd (%rcx){1to4}, %ymm29, %ymm24

// CHECK: vandpd 4064(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x54,0x42,0x7f]
          vandpd 4064(%rdx), %ymm29, %ymm24

// CHECK: vandpd 4096(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x54,0x82,0x00,0x10,0x00,0x00]
          vandpd 4096(%rdx), %ymm29, %ymm24

// CHECK: vandpd -4096(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x54,0x42,0x80]
          vandpd -4096(%rdx), %ymm29, %ymm24

// CHECK: vandpd -4128(%rdx), %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x54,0x82,0xe0,0xef,0xff,0xff]
          vandpd -4128(%rdx), %ymm29, %ymm24

// CHECK: vandpd 1016(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x54,0x42,0x7f]
          vandpd 1016(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vandpd 1024(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x54,0x82,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vandpd -1024(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x54,0x42,0x80]
          vandpd -1024(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vandpd -1032(%rdx){1to4}, %ymm29, %ymm24
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x54,0x82,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to4}, %ymm29, %ymm24

// CHECK: vandps %xmm17, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xa1,0x34,0x00,0x54,0xf1]
          vandps %xmm17, %xmm25, %xmm22

// CHECK: vandps %xmm17, %xmm25, %xmm22 {%k3}
// CHECK:  encoding: [0x62,0xa1,0x34,0x03,0x54,0xf1]
          vandps %xmm17, %xmm25, %xmm22 {%k3}

// CHECK: vandps %xmm17, %xmm25, %xmm22 {%k3} {z}
// CHECK:  encoding: [0x62,0xa1,0x34,0x83,0x54,0xf1]
          vandps %xmm17, %xmm25, %xmm22 {%k3} {z}

// CHECK: vandps (%rcx), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x00,0x54,0x31]
          vandps (%rcx), %xmm25, %xmm22

// CHECK: vandps 4660(%rax,%r14,8), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xa1,0x34,0x00,0x54,0xb4,0xf0,0x34,0x12,0x00,0x00]
          vandps 4660(%rax,%r14,8), %xmm25, %xmm22

// CHECK: vandps (%rcx){1to4}, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x10,0x54,0x31]
          vandps (%rcx){1to4}, %xmm25, %xmm22

// CHECK: vandps 2032(%rdx), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x00,0x54,0x72,0x7f]
          vandps 2032(%rdx), %xmm25, %xmm22

// CHECK: vandps 2048(%rdx), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x00,0x54,0xb2,0x00,0x08,0x00,0x00]
          vandps 2048(%rdx), %xmm25, %xmm22

// CHECK: vandps -2048(%rdx), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x00,0x54,0x72,0x80]
          vandps -2048(%rdx), %xmm25, %xmm22

// CHECK: vandps -2064(%rdx), %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x00,0x54,0xb2,0xf0,0xf7,0xff,0xff]
          vandps -2064(%rdx), %xmm25, %xmm22

// CHECK: vandps 508(%rdx){1to4}, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x10,0x54,0x72,0x7f]
          vandps 508(%rdx){1to4}, %xmm25, %xmm22

// CHECK: vandps 512(%rdx){1to4}, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x10,0x54,0xb2,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to4}, %xmm25, %xmm22

// CHECK: vandps -512(%rdx){1to4}, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x10,0x54,0x72,0x80]
          vandps -512(%rdx){1to4}, %xmm25, %xmm22

// CHECK: vandps -516(%rdx){1to4}, %xmm25, %xmm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x10,0x54,0xb2,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to4}, %xmm25, %xmm22

// CHECK: vandps %ymm18, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xa1,0x34,0x20,0x54,0xf2]
          vandps %ymm18, %ymm25, %ymm22

// CHECK: vandps %ymm18, %ymm25, %ymm22 {%k1}
// CHECK:  encoding: [0x62,0xa1,0x34,0x21,0x54,0xf2]
          vandps %ymm18, %ymm25, %ymm22 {%k1}

// CHECK: vandps %ymm18, %ymm25, %ymm22 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0x34,0xa1,0x54,0xf2]
          vandps %ymm18, %ymm25, %ymm22 {%k1} {z}

// CHECK: vandps (%rcx), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x20,0x54,0x31]
          vandps (%rcx), %ymm25, %ymm22

// CHECK: vandps 4660(%rax,%r14,8), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xa1,0x34,0x20,0x54,0xb4,0xf0,0x34,0x12,0x00,0x00]
          vandps 4660(%rax,%r14,8), %ymm25, %ymm22

// CHECK: vandps (%rcx){1to8}, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x30,0x54,0x31]
          vandps (%rcx){1to8}, %ymm25, %ymm22

// CHECK: vandps 4064(%rdx), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x20,0x54,0x72,0x7f]
          vandps 4064(%rdx), %ymm25, %ymm22

// CHECK: vandps 4096(%rdx), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x20,0x54,0xb2,0x00,0x10,0x00,0x00]
          vandps 4096(%rdx), %ymm25, %ymm22

// CHECK: vandps -4096(%rdx), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x20,0x54,0x72,0x80]
          vandps -4096(%rdx), %ymm25, %ymm22

// CHECK: vandps -4128(%rdx), %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x20,0x54,0xb2,0xe0,0xef,0xff,0xff]
          vandps -4128(%rdx), %ymm25, %ymm22

// CHECK: vandps 508(%rdx){1to8}, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x30,0x54,0x72,0x7f]
          vandps 508(%rdx){1to8}, %ymm25, %ymm22

// CHECK: vandps 512(%rdx){1to8}, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x30,0x54,0xb2,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to8}, %ymm25, %ymm22

// CHECK: vandps -512(%rdx){1to8}, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x30,0x54,0x72,0x80]
          vandps -512(%rdx){1to8}, %ymm25, %ymm22

// CHECK: vandps -516(%rdx){1to8}, %ymm25, %ymm22
// CHECK:  encoding: [0x62,0xe1,0x34,0x30,0x54,0xb2,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to8}, %ymm25, %ymm22

// CHECK: vandnpd %xmm23, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xa1,0xed,0x00,0x55,0xdf]
          vandnpd %xmm23, %xmm18, %xmm19

// CHECK: vandnpd %xmm23, %xmm18, %xmm19 {%k1}
// CHECK:  encoding: [0x62,0xa1,0xed,0x01,0x55,0xdf]
          vandnpd %xmm23, %xmm18, %xmm19 {%k1}

// CHECK: vandnpd %xmm23, %xmm18, %xmm19 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0xed,0x81,0x55,0xdf]
          vandnpd %xmm23, %xmm18, %xmm19 {%k1} {z}

// CHECK: vandnpd (%rcx), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x00,0x55,0x19]
          vandnpd (%rcx), %xmm18, %xmm19

// CHECK: vandnpd 4660(%rax,%r14,8), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xa1,0xed,0x00,0x55,0x9c,0xf0,0x34,0x12,0x00,0x00]
          vandnpd 4660(%rax,%r14,8), %xmm18, %xmm19

// CHECK: vandnpd (%rcx){1to2}, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x10,0x55,0x19]
          vandnpd (%rcx){1to2}, %xmm18, %xmm19

// CHECK: vandnpd 2032(%rdx), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x00,0x55,0x5a,0x7f]
          vandnpd 2032(%rdx), %xmm18, %xmm19

// CHECK: vandnpd 2048(%rdx), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x00,0x55,0x9a,0x00,0x08,0x00,0x00]
          vandnpd 2048(%rdx), %xmm18, %xmm19

// CHECK: vandnpd -2048(%rdx), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x00,0x55,0x5a,0x80]
          vandnpd -2048(%rdx), %xmm18, %xmm19

// CHECK: vandnpd -2064(%rdx), %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x00,0x55,0x9a,0xf0,0xf7,0xff,0xff]
          vandnpd -2064(%rdx), %xmm18, %xmm19

// CHECK: vandnpd 1016(%rdx){1to2}, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x10,0x55,0x5a,0x7f]
          vandnpd 1016(%rdx){1to2}, %xmm18, %xmm19

// CHECK: vandnpd 1024(%rdx){1to2}, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x10,0x55,0x9a,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to2}, %xmm18, %xmm19

// CHECK: vandnpd -1024(%rdx){1to2}, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x10,0x55,0x5a,0x80]
          vandnpd -1024(%rdx){1to2}, %xmm18, %xmm19

// CHECK: vandnpd -1032(%rdx){1to2}, %xmm18, %xmm19
// CHECK:  encoding: [0x62,0xe1,0xed,0x10,0x55,0x9a,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to2}, %xmm18, %xmm19

// CHECK: vandnpd %ymm28, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x01,0x95,0x20,0x55,0xcc]
          vandnpd %ymm28, %ymm29, %ymm25

// CHECK: vandnpd %ymm28, %ymm29, %ymm25 {%k7}
// CHECK:  encoding: [0x62,0x01,0x95,0x27,0x55,0xcc]
          vandnpd %ymm28, %ymm29, %ymm25 {%k7}

// CHECK: vandnpd %ymm28, %ymm29, %ymm25 {%k7} {z}
// CHECK:  encoding: [0x62,0x01,0x95,0xa7,0x55,0xcc]
          vandnpd %ymm28, %ymm29, %ymm25 {%k7} {z}

// CHECK: vandnpd (%rcx), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x55,0x09]
          vandnpd (%rcx), %ymm29, %ymm25

// CHECK: vandnpd 4660(%rax,%r14,8), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x21,0x95,0x20,0x55,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vandnpd 4660(%rax,%r14,8), %ymm29, %ymm25

// CHECK: vandnpd (%rcx){1to4}, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x55,0x09]
          vandnpd (%rcx){1to4}, %ymm29, %ymm25

// CHECK: vandnpd 4064(%rdx), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x55,0x4a,0x7f]
          vandnpd 4064(%rdx), %ymm29, %ymm25

// CHECK: vandnpd 4096(%rdx), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x55,0x8a,0x00,0x10,0x00,0x00]
          vandnpd 4096(%rdx), %ymm29, %ymm25

// CHECK: vandnpd -4096(%rdx), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x55,0x4a,0x80]
          vandnpd -4096(%rdx), %ymm29, %ymm25

// CHECK: vandnpd -4128(%rdx), %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x20,0x55,0x8a,0xe0,0xef,0xff,0xff]
          vandnpd -4128(%rdx), %ymm29, %ymm25

// CHECK: vandnpd 1016(%rdx){1to4}, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x55,0x4a,0x7f]
          vandnpd 1016(%rdx){1to4}, %ymm29, %ymm25

// CHECK: vandnpd 1024(%rdx){1to4}, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x55,0x8a,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to4}, %ymm29, %ymm25

// CHECK: vandnpd -1024(%rdx){1to4}, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x55,0x4a,0x80]
          vandnpd -1024(%rdx){1to4}, %ymm29, %ymm25

// CHECK: vandnpd -1032(%rdx){1to4}, %ymm29, %ymm25
// CHECK:  encoding: [0x62,0x61,0x95,0x30,0x55,0x8a,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to4}, %ymm29, %ymm25

// CHECK: vandnps %xmm23, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x21,0x74,0x00,0x55,0xcf]
          vandnps %xmm23, %xmm17, %xmm25

// CHECK: vandnps %xmm23, %xmm17, %xmm25 {%k5}
// CHECK:  encoding: [0x62,0x21,0x74,0x05,0x55,0xcf]
          vandnps %xmm23, %xmm17, %xmm25 {%k5}

// CHECK: vandnps %xmm23, %xmm17, %xmm25 {%k5} {z}
// CHECK:  encoding: [0x62,0x21,0x74,0x85,0x55,0xcf]
          vandnps %xmm23, %xmm17, %xmm25 {%k5} {z}

// CHECK: vandnps (%rcx), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x00,0x55,0x09]
          vandnps (%rcx), %xmm17, %xmm25

// CHECK: vandnps 4660(%rax,%r14,8), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x21,0x74,0x00,0x55,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vandnps 4660(%rax,%r14,8), %xmm17, %xmm25

// CHECK: vandnps (%rcx){1to4}, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x10,0x55,0x09]
          vandnps (%rcx){1to4}, %xmm17, %xmm25

// CHECK: vandnps 2032(%rdx), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x00,0x55,0x4a,0x7f]
          vandnps 2032(%rdx), %xmm17, %xmm25

// CHECK: vandnps 2048(%rdx), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x00,0x55,0x8a,0x00,0x08,0x00,0x00]
          vandnps 2048(%rdx), %xmm17, %xmm25

// CHECK: vandnps -2048(%rdx), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x00,0x55,0x4a,0x80]
          vandnps -2048(%rdx), %xmm17, %xmm25

// CHECK: vandnps -2064(%rdx), %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x00,0x55,0x8a,0xf0,0xf7,0xff,0xff]
          vandnps -2064(%rdx), %xmm17, %xmm25

// CHECK: vandnps 508(%rdx){1to4}, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x10,0x55,0x4a,0x7f]
          vandnps 508(%rdx){1to4}, %xmm17, %xmm25

// CHECK: vandnps 512(%rdx){1to4}, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x10,0x55,0x8a,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to4}, %xmm17, %xmm25

// CHECK: vandnps -512(%rdx){1to4}, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x10,0x55,0x4a,0x80]
          vandnps -512(%rdx){1to4}, %xmm17, %xmm25

// CHECK: vandnps -516(%rdx){1to4}, %xmm17, %xmm25
// CHECK:  encoding: [0x62,0x61,0x74,0x10,0x55,0x8a,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to4}, %xmm17, %xmm25

// CHECK: vandnps %ymm23, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xa1,0x64,0x20,0x55,0xd7]
          vandnps %ymm23, %ymm19, %ymm18

// CHECK: vandnps %ymm23, %ymm19, %ymm18 {%k6}
// CHECK:  encoding: [0x62,0xa1,0x64,0x26,0x55,0xd7]
          vandnps %ymm23, %ymm19, %ymm18 {%k6}

// CHECK: vandnps %ymm23, %ymm19, %ymm18 {%k6} {z}
// CHECK:  encoding: [0x62,0xa1,0x64,0xa6,0x55,0xd7]
          vandnps %ymm23, %ymm19, %ymm18 {%k6} {z}

// CHECK: vandnps (%rcx), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x20,0x55,0x11]
          vandnps (%rcx), %ymm19, %ymm18

// CHECK: vandnps 4660(%rax,%r14,8), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xa1,0x64,0x20,0x55,0x94,0xf0,0x34,0x12,0x00,0x00]
          vandnps 4660(%rax,%r14,8), %ymm19, %ymm18

// CHECK: vandnps (%rcx){1to8}, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x30,0x55,0x11]
          vandnps (%rcx){1to8}, %ymm19, %ymm18

// CHECK: vandnps 4064(%rdx), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x20,0x55,0x52,0x7f]
          vandnps 4064(%rdx), %ymm19, %ymm18

// CHECK: vandnps 4096(%rdx), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x20,0x55,0x92,0x00,0x10,0x00,0x00]
          vandnps 4096(%rdx), %ymm19, %ymm18

// CHECK: vandnps -4096(%rdx), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x20,0x55,0x52,0x80]
          vandnps -4096(%rdx), %ymm19, %ymm18

// CHECK: vandnps -4128(%rdx), %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x20,0x55,0x92,0xe0,0xef,0xff,0xff]
          vandnps -4128(%rdx), %ymm19, %ymm18

// CHECK: vandnps 508(%rdx){1to8}, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x30,0x55,0x52,0x7f]
          vandnps 508(%rdx){1to8}, %ymm19, %ymm18

// CHECK: vandnps 512(%rdx){1to8}, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x30,0x55,0x92,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to8}, %ymm19, %ymm18

// CHECK: vandnps -512(%rdx){1to8}, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x30,0x55,0x52,0x80]
          vandnps -512(%rdx){1to8}, %ymm19, %ymm18

// CHECK: vandnps -516(%rdx){1to8}, %ymm19, %ymm18
// CHECK:  encoding: [0x62,0xe1,0x64,0x30,0x55,0x92,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to8}, %ymm19, %ymm18

// CHECK: vorpd  %xmm18, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x21,0x95,0x00,0x56,0xd2]
          vorpd  %xmm18, %xmm29, %xmm26

// CHECK: vorpd  %xmm18, %xmm29, %xmm26 {%k2}
// CHECK:  encoding: [0x62,0x21,0x95,0x02,0x56,0xd2]
          vorpd  %xmm18, %xmm29, %xmm26 {%k2}

// CHECK: vorpd  %xmm18, %xmm29, %xmm26 {%k2} {z}
// CHECK:  encoding: [0x62,0x21,0x95,0x82,0x56,0xd2]
          vorpd  %xmm18, %xmm29, %xmm26 {%k2} {z}

// CHECK: vorpd  (%rcx), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x00,0x56,0x11]
          vorpd  (%rcx), %xmm29, %xmm26

// CHECK: vorpd  4660(%rax,%r14,8), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x21,0x95,0x00,0x56,0x94,0xf0,0x34,0x12,0x00,0x00]
          vorpd  4660(%rax,%r14,8), %xmm29, %xmm26

// CHECK: vorpd  (%rcx){1to2}, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x10,0x56,0x11]
          vorpd  (%rcx){1to2}, %xmm29, %xmm26

// CHECK: vorpd  2032(%rdx), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x00,0x56,0x52,0x7f]
          vorpd  2032(%rdx), %xmm29, %xmm26

// CHECK: vorpd  2048(%rdx), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x00,0x56,0x92,0x00,0x08,0x00,0x00]
          vorpd  2048(%rdx), %xmm29, %xmm26

// CHECK: vorpd  -2048(%rdx), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x00,0x56,0x52,0x80]
          vorpd  -2048(%rdx), %xmm29, %xmm26

// CHECK: vorpd  -2064(%rdx), %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x00,0x56,0x92,0xf0,0xf7,0xff,0xff]
          vorpd  -2064(%rdx), %xmm29, %xmm26

// CHECK: vorpd  1016(%rdx){1to2}, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x10,0x56,0x52,0x7f]
          vorpd  1016(%rdx){1to2}, %xmm29, %xmm26

// CHECK: vorpd  1024(%rdx){1to2}, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x10,0x56,0x92,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to2}, %xmm29, %xmm26

// CHECK: vorpd  -1024(%rdx){1to2}, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x10,0x56,0x52,0x80]
          vorpd  -1024(%rdx){1to2}, %xmm29, %xmm26

// CHECK: vorpd  -1032(%rdx){1to2}, %xmm29, %xmm26
// CHECK:  encoding: [0x62,0x61,0x95,0x10,0x56,0x92,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to2}, %xmm29, %xmm26

// CHECK: vorpd  %ymm22, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x21,0xe5,0x20,0x56,0xe6]
          vorpd  %ymm22, %ymm19, %ymm28

// CHECK: vorpd  %ymm22, %ymm19, %ymm28 {%k1}
// CHECK:  encoding: [0x62,0x21,0xe5,0x21,0x56,0xe6]
          vorpd  %ymm22, %ymm19, %ymm28 {%k1}

// CHECK: vorpd  %ymm22, %ymm19, %ymm28 {%k1} {z}
// CHECK:  encoding: [0x62,0x21,0xe5,0xa1,0x56,0xe6]
          vorpd  %ymm22, %ymm19, %ymm28 {%k1} {z}

// CHECK: vorpd  (%rcx), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x20,0x56,0x21]
          vorpd  (%rcx), %ymm19, %ymm28

// CHECK: vorpd  4660(%rax,%r14,8), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x21,0xe5,0x20,0x56,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vorpd  4660(%rax,%r14,8), %ymm19, %ymm28

// CHECK: vorpd  (%rcx){1to4}, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x30,0x56,0x21]
          vorpd  (%rcx){1to4}, %ymm19, %ymm28

// CHECK: vorpd  4064(%rdx), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x20,0x56,0x62,0x7f]
          vorpd  4064(%rdx), %ymm19, %ymm28

// CHECK: vorpd  4096(%rdx), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x20,0x56,0xa2,0x00,0x10,0x00,0x00]
          vorpd  4096(%rdx), %ymm19, %ymm28

// CHECK: vorpd  -4096(%rdx), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x20,0x56,0x62,0x80]
          vorpd  -4096(%rdx), %ymm19, %ymm28

// CHECK: vorpd  -4128(%rdx), %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x20,0x56,0xa2,0xe0,0xef,0xff,0xff]
          vorpd  -4128(%rdx), %ymm19, %ymm28

// CHECK: vorpd  1016(%rdx){1to4}, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x30,0x56,0x62,0x7f]
          vorpd  1016(%rdx){1to4}, %ymm19, %ymm28

// CHECK: vorpd  1024(%rdx){1to4}, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x30,0x56,0xa2,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to4}, %ymm19, %ymm28

// CHECK: vorpd  -1024(%rdx){1to4}, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x30,0x56,0x62,0x80]
          vorpd  -1024(%rdx){1to4}, %ymm19, %ymm28

// CHECK: vorpd  -1032(%rdx){1to4}, %ymm19, %ymm28
// CHECK:  encoding: [0x62,0x61,0xe5,0x30,0x56,0xa2,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to4}, %ymm19, %ymm28

// CHECK: vorps  %xmm24, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x01,0x4c,0x00,0x56,0xe0]
          vorps  %xmm24, %xmm22, %xmm28

// CHECK: vorps  %xmm24, %xmm22, %xmm28 {%k6}
// CHECK:  encoding: [0x62,0x01,0x4c,0x06,0x56,0xe0]
          vorps  %xmm24, %xmm22, %xmm28 {%k6}

// CHECK: vorps  %xmm24, %xmm22, %xmm28 {%k6} {z}
// CHECK:  encoding: [0x62,0x01,0x4c,0x86,0x56,0xe0]
          vorps  %xmm24, %xmm22, %xmm28 {%k6} {z}

// CHECK: vorps  (%rcx), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x00,0x56,0x21]
          vorps  (%rcx), %xmm22, %xmm28

// CHECK: vorps  4660(%rax,%r14,8), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x21,0x4c,0x00,0x56,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vorps  4660(%rax,%r14,8), %xmm22, %xmm28

// CHECK: vorps  (%rcx){1to4}, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x10,0x56,0x21]
          vorps  (%rcx){1to4}, %xmm22, %xmm28

// CHECK: vorps  2032(%rdx), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x00,0x56,0x62,0x7f]
          vorps  2032(%rdx), %xmm22, %xmm28

// CHECK: vorps  2048(%rdx), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x00,0x56,0xa2,0x00,0x08,0x00,0x00]
          vorps  2048(%rdx), %xmm22, %xmm28

// CHECK: vorps  -2048(%rdx), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x00,0x56,0x62,0x80]
          vorps  -2048(%rdx), %xmm22, %xmm28

// CHECK: vorps  -2064(%rdx), %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x00,0x56,0xa2,0xf0,0xf7,0xff,0xff]
          vorps  -2064(%rdx), %xmm22, %xmm28

// CHECK: vorps  508(%rdx){1to4}, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x10,0x56,0x62,0x7f]
          vorps  508(%rdx){1to4}, %xmm22, %xmm28

// CHECK: vorps  512(%rdx){1to4}, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x10,0x56,0xa2,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to4}, %xmm22, %xmm28

// CHECK: vorps  -512(%rdx){1to4}, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x10,0x56,0x62,0x80]
          vorps  -512(%rdx){1to4}, %xmm22, %xmm28

// CHECK: vorps  -516(%rdx){1to4}, %xmm22, %xmm28
// CHECK:  encoding: [0x62,0x61,0x4c,0x10,0x56,0xa2,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to4}, %xmm22, %xmm28

// CHECK: vorps  %ymm25, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0x81,0x3c,0x20,0x56,0xe1]
          vorps  %ymm25, %ymm24, %ymm20

// CHECK: vorps  %ymm25, %ymm24, %ymm20 {%k1}
// CHECK:  encoding: [0x62,0x81,0x3c,0x21,0x56,0xe1]
          vorps  %ymm25, %ymm24, %ymm20 {%k1}

// CHECK: vorps  %ymm25, %ymm24, %ymm20 {%k1} {z}
// CHECK:  encoding: [0x62,0x81,0x3c,0xa1,0x56,0xe1]
          vorps  %ymm25, %ymm24, %ymm20 {%k1} {z}

// CHECK: vorps  (%rcx), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x20,0x56,0x21]
          vorps  (%rcx), %ymm24, %ymm20

// CHECK: vorps  4660(%rax,%r14,8), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xa1,0x3c,0x20,0x56,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vorps  4660(%rax,%r14,8), %ymm24, %ymm20

// CHECK: vorps  (%rcx){1to8}, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x30,0x56,0x21]
          vorps  (%rcx){1to8}, %ymm24, %ymm20

// CHECK: vorps  4064(%rdx), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x20,0x56,0x62,0x7f]
          vorps  4064(%rdx), %ymm24, %ymm20

// CHECK: vorps  4096(%rdx), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x20,0x56,0xa2,0x00,0x10,0x00,0x00]
          vorps  4096(%rdx), %ymm24, %ymm20

// CHECK: vorps  -4096(%rdx), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x20,0x56,0x62,0x80]
          vorps  -4096(%rdx), %ymm24, %ymm20

// CHECK: vorps  -4128(%rdx), %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x20,0x56,0xa2,0xe0,0xef,0xff,0xff]
          vorps  -4128(%rdx), %ymm24, %ymm20

// CHECK: vorps  508(%rdx){1to8}, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x30,0x56,0x62,0x7f]
          vorps  508(%rdx){1to8}, %ymm24, %ymm20

// CHECK: vorps  512(%rdx){1to8}, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x30,0x56,0xa2,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to8}, %ymm24, %ymm20

// CHECK: vorps  -512(%rdx){1to8}, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x30,0x56,0x62,0x80]
          vorps  -512(%rdx){1to8}, %ymm24, %ymm20

// CHECK: vorps  -516(%rdx){1to8}, %ymm24, %ymm20
// CHECK:  encoding: [0x62,0xe1,0x3c,0x30,0x56,0xa2,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to8}, %ymm24, %ymm20

// CHECK: vxorpd %xmm18, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xa1,0xd5,0x00,0x57,0xf2]
          vxorpd %xmm18, %xmm21, %xmm22

// CHECK: vxorpd %xmm18, %xmm21, %xmm22 {%k3}
// CHECK:  encoding: [0x62,0xa1,0xd5,0x03,0x57,0xf2]
          vxorpd %xmm18, %xmm21, %xmm22 {%k3}

// CHECK: vxorpd %xmm18, %xmm21, %xmm22 {%k3} {z}
// CHECK:  encoding: [0x62,0xa1,0xd5,0x83,0x57,0xf2]
          vxorpd %xmm18, %xmm21, %xmm22 {%k3} {z}

// CHECK: vxorpd (%rcx), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x31]
          vxorpd (%rcx), %xmm21, %xmm22

// CHECK: vxorpd 4660(%rax,%r14,8), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xa1,0xd5,0x00,0x57,0xb4,0xf0,0x34,0x12,0x00,0x00]
          vxorpd 4660(%rax,%r14,8), %xmm21, %xmm22

// CHECK: vxorpd (%rcx){1to2}, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x31]
          vxorpd (%rcx){1to2}, %xmm21, %xmm22

// CHECK: vxorpd 2032(%rdx), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x72,0x7f]
          vxorpd 2032(%rdx), %xmm21, %xmm22

// CHECK: vxorpd 2048(%rdx), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0xb2,0x00,0x08,0x00,0x00]
          vxorpd 2048(%rdx), %xmm21, %xmm22

// CHECK: vxorpd -2048(%rdx), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0x72,0x80]
          vxorpd -2048(%rdx), %xmm21, %xmm22

// CHECK: vxorpd -2064(%rdx), %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x00,0x57,0xb2,0xf0,0xf7,0xff,0xff]
          vxorpd -2064(%rdx), %xmm21, %xmm22

// CHECK: vxorpd 1016(%rdx){1to2}, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x72,0x7f]
          vxorpd 1016(%rdx){1to2}, %xmm21, %xmm22

// CHECK: vxorpd 1024(%rdx){1to2}, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0xb2,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to2}, %xmm21, %xmm22

// CHECK: vxorpd -1024(%rdx){1to2}, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0x72,0x80]
          vxorpd -1024(%rdx){1to2}, %xmm21, %xmm22

// CHECK: vxorpd -1032(%rdx){1to2}, %xmm21, %xmm22
// CHECK:  encoding: [0x62,0xe1,0xd5,0x10,0x57,0xb2,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to2}, %xmm21, %xmm22

// CHECK: vxorpd %ymm27, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x01,0xd5,0x20,0x57,0xcb]
          vxorpd %ymm27, %ymm21, %ymm25

// CHECK: vxorpd %ymm27, %ymm21, %ymm25 {%k7}
// CHECK:  encoding: [0x62,0x01,0xd5,0x27,0x57,0xcb]
          vxorpd %ymm27, %ymm21, %ymm25 {%k7}

// CHECK: vxorpd %ymm27, %ymm21, %ymm25 {%k7} {z}
// CHECK:  encoding: [0x62,0x01,0xd5,0xa7,0x57,0xcb]
          vxorpd %ymm27, %ymm21, %ymm25 {%k7} {z}

// CHECK: vxorpd (%rcx), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x57,0x09]
          vxorpd (%rcx), %ymm21, %ymm25

// CHECK: vxorpd 4660(%rax,%r14,8), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x21,0xd5,0x20,0x57,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vxorpd 4660(%rax,%r14,8), %ymm21, %ymm25

// CHECK: vxorpd (%rcx){1to4}, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x57,0x09]
          vxorpd (%rcx){1to4}, %ymm21, %ymm25

// CHECK: vxorpd 4064(%rdx), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x57,0x4a,0x7f]
          vxorpd 4064(%rdx), %ymm21, %ymm25

// CHECK: vxorpd 4096(%rdx), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x57,0x8a,0x00,0x10,0x00,0x00]
          vxorpd 4096(%rdx), %ymm21, %ymm25

// CHECK: vxorpd -4096(%rdx), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x57,0x4a,0x80]
          vxorpd -4096(%rdx), %ymm21, %ymm25

// CHECK: vxorpd -4128(%rdx), %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x20,0x57,0x8a,0xe0,0xef,0xff,0xff]
          vxorpd -4128(%rdx), %ymm21, %ymm25

// CHECK: vxorpd 1016(%rdx){1to4}, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x57,0x4a,0x7f]
          vxorpd 1016(%rdx){1to4}, %ymm21, %ymm25

// CHECK: vxorpd 1024(%rdx){1to4}, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x57,0x8a,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to4}, %ymm21, %ymm25

// CHECK: vxorpd -1024(%rdx){1to4}, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x57,0x4a,0x80]
          vxorpd -1024(%rdx){1to4}, %ymm21, %ymm25

// CHECK: vxorpd -1032(%rdx){1to4}, %ymm21, %ymm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x30,0x57,0x8a,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to4}, %ymm21, %ymm25

// CHECK: vxorps %xmm21, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xa1,0x54,0x00,0x57,0xcd]
          vxorps %xmm21, %xmm21, %xmm17

// CHECK: vxorps %xmm21, %xmm21, %xmm17 {%k5}
// CHECK:  encoding: [0x62,0xa1,0x54,0x05,0x57,0xcd]
          vxorps %xmm21, %xmm21, %xmm17 {%k5}

// CHECK: vxorps %xmm21, %xmm21, %xmm17 {%k5} {z}
// CHECK:  encoding: [0x62,0xa1,0x54,0x85,0x57,0xcd]
          vxorps %xmm21, %xmm21, %xmm17 {%k5} {z}

// CHECK: vxorps (%rcx), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x57,0x09]
          vxorps (%rcx), %xmm21, %xmm17

// CHECK: vxorps 4660(%rax,%r14,8), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xa1,0x54,0x00,0x57,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vxorps 4660(%rax,%r14,8), %xmm21, %xmm17

// CHECK: vxorps (%rcx){1to4}, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x57,0x09]
          vxorps (%rcx){1to4}, %xmm21, %xmm17

// CHECK: vxorps 2032(%rdx), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x57,0x4a,0x7f]
          vxorps 2032(%rdx), %xmm21, %xmm17

// CHECK: vxorps 2048(%rdx), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x57,0x8a,0x00,0x08,0x00,0x00]
          vxorps 2048(%rdx), %xmm21, %xmm17

// CHECK: vxorps -2048(%rdx), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x57,0x4a,0x80]
          vxorps -2048(%rdx), %xmm21, %xmm17

// CHECK: vxorps -2064(%rdx), %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x00,0x57,0x8a,0xf0,0xf7,0xff,0xff]
          vxorps -2064(%rdx), %xmm21, %xmm17

// CHECK: vxorps 508(%rdx){1to4}, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x57,0x4a,0x7f]
          vxorps 508(%rdx){1to4}, %xmm21, %xmm17

// CHECK: vxorps 512(%rdx){1to4}, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x57,0x8a,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to4}, %xmm21, %xmm17

// CHECK: vxorps -512(%rdx){1to4}, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x57,0x4a,0x80]
          vxorps -512(%rdx){1to4}, %xmm21, %xmm17

// CHECK: vxorps -516(%rdx){1to4}, %xmm21, %xmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x10,0x57,0x8a,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to4}, %xmm21, %xmm17

// CHECK: vxorps %ymm22, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x21,0x34,0x20,0x57,0xe6]
          vxorps %ymm22, %ymm25, %ymm28

// CHECK: vxorps %ymm22, %ymm25, %ymm28 {%k3}
// CHECK:  encoding: [0x62,0x21,0x34,0x23,0x57,0xe6]
          vxorps %ymm22, %ymm25, %ymm28 {%k3}

// CHECK: vxorps %ymm22, %ymm25, %ymm28 {%k3} {z}
// CHECK:  encoding: [0x62,0x21,0x34,0xa3,0x57,0xe6]
          vxorps %ymm22, %ymm25, %ymm28 {%k3} {z}

// CHECK: vxorps (%rcx), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x20,0x57,0x21]
          vxorps (%rcx), %ymm25, %ymm28

// CHECK: vxorps 4660(%rax,%r14,8), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x21,0x34,0x20,0x57,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vxorps 4660(%rax,%r14,8), %ymm25, %ymm28

// CHECK: vxorps (%rcx){1to8}, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x30,0x57,0x21]
          vxorps (%rcx){1to8}, %ymm25, %ymm28

// CHECK: vxorps 4064(%rdx), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x20,0x57,0x62,0x7f]
          vxorps 4064(%rdx), %ymm25, %ymm28

// CHECK: vxorps 4096(%rdx), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x20,0x57,0xa2,0x00,0x10,0x00,0x00]
          vxorps 4096(%rdx), %ymm25, %ymm28

// CHECK: vxorps -4096(%rdx), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x20,0x57,0x62,0x80]
          vxorps -4096(%rdx), %ymm25, %ymm28

// CHECK: vxorps -4128(%rdx), %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x20,0x57,0xa2,0xe0,0xef,0xff,0xff]
          vxorps -4128(%rdx), %ymm25, %ymm28

// CHECK: vxorps 508(%rdx){1to8}, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x30,0x57,0x62,0x7f]
          vxorps 508(%rdx){1to8}, %ymm25, %ymm28

// CHECK: vxorps 512(%rdx){1to8}, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x30,0x57,0xa2,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to8}, %ymm25, %ymm28

// CHECK: vxorps -512(%rdx){1to8}, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x30,0x57,0x62,0x80]
          vxorps -512(%rdx){1to8}, %ymm25, %ymm28

// CHECK: vxorps -516(%rdx){1to8}, %ymm25, %ymm28
// CHECK:  encoding: [0x62,0x61,0x34,0x30,0x57,0xa2,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to8}, %ymm25, %ymm28

// CHECK: vbroadcastf64x2 (%rcx), %ymm27
// CHECK:  encoding: [0x62,0x62,0xfd,0x28,0x1a,0x19]
          vbroadcastf64x2 (%rcx), %ymm27

// CHECK: vbroadcastf64x2 (%rcx), %ymm27 {%k5}
// CHECK:  encoding: [0x62,0x62,0xfd,0x2d,0x1a,0x19]
          vbroadcastf64x2 (%rcx), %ymm27 {%k5}

// CHECK: vbroadcastf64x2 (%rcx), %ymm27 {%k5} {z}
// CHECK:  encoding: [0x62,0x62,0xfd,0xad,0x1a,0x19]
          vbroadcastf64x2 (%rcx), %ymm27 {%k5} {z}

// CHECK: vbroadcastf64x2 291(%rax,%r14,8), %ymm27
// CHECK:  encoding: [0x62,0x22,0xfd,0x28,0x1a,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vbroadcastf64x2 291(%rax,%r14,8), %ymm27

// CHECK: vbroadcastf64x2 2032(%rdx), %ymm27
// CHECK:  encoding: [0x62,0x62,0xfd,0x28,0x1a,0x5a,0x7f]
          vbroadcastf64x2 2032(%rdx), %ymm27

// CHECK: vbroadcastf64x2 2048(%rdx), %ymm27
// CHECK:  encoding: [0x62,0x62,0xfd,0x28,0x1a,0x9a,0x00,0x08,0x00,0x00]
          vbroadcastf64x2 2048(%rdx), %ymm27

// CHECK: vbroadcastf64x2 -2048(%rdx), %ymm27
// CHECK:  encoding: [0x62,0x62,0xfd,0x28,0x1a,0x5a,0x80]
          vbroadcastf64x2 -2048(%rdx), %ymm27

// CHECK: vbroadcastf64x2 -2064(%rdx), %ymm27
// CHECK:  encoding: [0x62,0x62,0xfd,0x28,0x1a,0x9a,0xf0,0xf7,0xff,0xff]
          vbroadcastf64x2 -2064(%rdx), %ymm27

// CHECK: vbroadcasti64x2 (%rcx), %ymm18
// CHECK:  encoding: [0x62,0xe2,0xfd,0x28,0x5a,0x11]
          vbroadcasti64x2 (%rcx), %ymm18

// CHECK: vbroadcasti64x2 (%rcx), %ymm18 {%k1}
// CHECK:  encoding: [0x62,0xe2,0xfd,0x29,0x5a,0x11]
          vbroadcasti64x2 (%rcx), %ymm18 {%k1}

// CHECK: vbroadcasti64x2 (%rcx), %ymm18 {%k1} {z}
// CHECK:  encoding: [0x62,0xe2,0xfd,0xa9,0x5a,0x11]
          vbroadcasti64x2 (%rcx), %ymm18 {%k1} {z}

// CHECK: vbroadcasti64x2 291(%rax,%r14,8), %ymm18
// CHECK:  encoding: [0x62,0xa2,0xfd,0x28,0x5a,0x94,0xf0,0x23,0x01,0x00,0x00]
          vbroadcasti64x2 291(%rax,%r14,8), %ymm18

// CHECK: vbroadcasti64x2 2032(%rdx), %ymm18
// CHECK:  encoding: [0x62,0xe2,0xfd,0x28,0x5a,0x52,0x7f]
          vbroadcasti64x2 2032(%rdx), %ymm18

// CHECK: vbroadcasti64x2 2048(%rdx), %ymm18
// CHECK:  encoding: [0x62,0xe2,0xfd,0x28,0x5a,0x92,0x00,0x08,0x00,0x00]
          vbroadcasti64x2 2048(%rdx), %ymm18

// CHECK: vbroadcasti64x2 -2048(%rdx), %ymm18
// CHECK:  encoding: [0x62,0xe2,0xfd,0x28,0x5a,0x52,0x80]
          vbroadcasti64x2 -2048(%rdx), %ymm18

// CHECK: vbroadcasti64x2 -2064(%rdx), %ymm18
// CHECK:  encoding: [0x62,0xe2,0xfd,0x28,0x5a,0x92,0xf0,0xf7,0xff,0xff]
          vbroadcasti64x2 -2064(%rdx), %ymm18

