// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx  --show-encoding %s | FileCheck %s

// CHECK: vblendmpd %xmm19, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x22,0xdd,0x00,0x65,0xdb]
          vblendmpd %xmm19, %xmm20, %xmm27

// CHECK: vblendmpd %xmm19, %xmm20, %xmm27 {%k7}
// CHECK:  encoding: [0x62,0x22,0xdd,0x07,0x65,0xdb]
          vblendmpd %xmm19, %xmm20, %xmm27 {%k7}

// CHECK: vblendmpd %xmm19, %xmm20, %xmm27 {%k7} {z}
// CHECK:  encoding: [0x62,0x22,0xdd,0x87,0x65,0xdb]
          vblendmpd %xmm19, %xmm20, %xmm27 {%k7} {z}

// CHECK: vblendmpd (%rcx), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x00,0x65,0x19]
          vblendmpd (%rcx), %xmm20, %xmm27

// CHECK: vblendmpd 291(%rax,%r14,8), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x22,0xdd,0x00,0x65,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vblendmpd 291(%rax,%r14,8), %xmm20, %xmm27

// CHECK: vblendmpd (%rcx){1to2}, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x10,0x65,0x19]
          vblendmpd (%rcx){1to2}, %xmm20, %xmm27

// CHECK: vblendmpd 2032(%rdx), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x00,0x65,0x5a,0x7f]
          vblendmpd 2032(%rdx), %xmm20, %xmm27

// CHECK: vblendmpd 2048(%rdx), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x00,0x65,0x9a,0x00,0x08,0x00,0x00]
          vblendmpd 2048(%rdx), %xmm20, %xmm27

// CHECK: vblendmpd -2048(%rdx), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x00,0x65,0x5a,0x80]
          vblendmpd -2048(%rdx), %xmm20, %xmm27

// CHECK: vblendmpd -2064(%rdx), %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x00,0x65,0x9a,0xf0,0xf7,0xff,0xff]
          vblendmpd -2064(%rdx), %xmm20, %xmm27

// CHECK: vblendmpd 1016(%rdx){1to2}, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x10,0x65,0x5a,0x7f]
          vblendmpd 1016(%rdx){1to2}, %xmm20, %xmm27

// CHECK: vblendmpd 1024(%rdx){1to2}, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x10,0x65,0x9a,0x00,0x04,0x00,0x00]
          vblendmpd 1024(%rdx){1to2}, %xmm20, %xmm27

// CHECK: vblendmpd -1024(%rdx){1to2}, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x10,0x65,0x5a,0x80]
          vblendmpd -1024(%rdx){1to2}, %xmm20, %xmm27

// CHECK: vblendmpd -1032(%rdx){1to2}, %xmm20, %xmm27
// CHECK:  encoding: [0x62,0x62,0xdd,0x10,0x65,0x9a,0xf8,0xfb,0xff,0xff]
          vblendmpd -1032(%rdx){1to2}, %xmm20, %xmm27

// CHECK: vblendmpd %ymm23, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x22,0xd5,0x20,0x65,0xe7]
          vblendmpd %ymm23, %ymm21, %ymm28

// CHECK: vblendmpd %ymm23, %ymm21, %ymm28 {%k3}
// CHECK:  encoding: [0x62,0x22,0xd5,0x23,0x65,0xe7]
          vblendmpd %ymm23, %ymm21, %ymm28 {%k3}

// CHECK: vblendmpd %ymm23, %ymm21, %ymm28 {%k3} {z}
// CHECK:  encoding: [0x62,0x22,0xd5,0xa3,0x65,0xe7]
          vblendmpd %ymm23, %ymm21, %ymm28 {%k3} {z}

// CHECK: vblendmpd (%rcx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x20,0x65,0x21]
          vblendmpd (%rcx), %ymm21, %ymm28

// CHECK: vblendmpd 291(%rax,%r14,8), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x22,0xd5,0x20,0x65,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vblendmpd 291(%rax,%r14,8), %ymm21, %ymm28

// CHECK: vblendmpd (%rcx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x30,0x65,0x21]
          vblendmpd (%rcx){1to4}, %ymm21, %ymm28

// CHECK: vblendmpd 4064(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x20,0x65,0x62,0x7f]
          vblendmpd 4064(%rdx), %ymm21, %ymm28

// CHECK: vblendmpd 4096(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x20,0x65,0xa2,0x00,0x10,0x00,0x00]
          vblendmpd 4096(%rdx), %ymm21, %ymm28

// CHECK: vblendmpd -4096(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x20,0x65,0x62,0x80]
          vblendmpd -4096(%rdx), %ymm21, %ymm28

// CHECK: vblendmpd -4128(%rdx), %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x20,0x65,0xa2,0xe0,0xef,0xff,0xff]
          vblendmpd -4128(%rdx), %ymm21, %ymm28

// CHECK: vblendmpd 1016(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x30,0x65,0x62,0x7f]
          vblendmpd 1016(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vblendmpd 1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x30,0x65,0xa2,0x00,0x04,0x00,0x00]
          vblendmpd 1024(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vblendmpd -1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x30,0x65,0x62,0x80]
          vblendmpd -1024(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vblendmpd -1032(%rdx){1to4}, %ymm21, %ymm28
// CHECK:  encoding: [0x62,0x62,0xd5,0x30,0x65,0xa2,0xf8,0xfb,0xff,0xff]
          vblendmpd -1032(%rdx){1to4}, %ymm21, %ymm28

// CHECK: vblendmps %xmm20, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x22,0x5d,0x00,0x65,0xc4]
          vblendmps %xmm20, %xmm20, %xmm24

// CHECK: vblendmps %xmm20, %xmm20, %xmm24 {%k1}
// CHECK:  encoding: [0x62,0x22,0x5d,0x01,0x65,0xc4]
          vblendmps %xmm20, %xmm20, %xmm24 {%k1}

// CHECK: vblendmps %xmm20, %xmm20, %xmm24 {%k1} {z}
// CHECK:  encoding: [0x62,0x22,0x5d,0x81,0x65,0xc4]
          vblendmps %xmm20, %xmm20, %xmm24 {%k1} {z}

// CHECK: vblendmps (%rcx), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x00,0x65,0x01]
          vblendmps (%rcx), %xmm20, %xmm24

// CHECK: vblendmps 291(%rax,%r14,8), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x22,0x5d,0x00,0x65,0x84,0xf0,0x23,0x01,0x00,0x00]
          vblendmps 291(%rax,%r14,8), %xmm20, %xmm24

// CHECK: vblendmps (%rcx){1to4}, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x10,0x65,0x01]
          vblendmps (%rcx){1to4}, %xmm20, %xmm24

// CHECK: vblendmps 2032(%rdx), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x00,0x65,0x42,0x7f]
          vblendmps 2032(%rdx), %xmm20, %xmm24

// CHECK: vblendmps 2048(%rdx), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x00,0x65,0x82,0x00,0x08,0x00,0x00]
          vblendmps 2048(%rdx), %xmm20, %xmm24

// CHECK: vblendmps -2048(%rdx), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x00,0x65,0x42,0x80]
          vblendmps -2048(%rdx), %xmm20, %xmm24

// CHECK: vblendmps -2064(%rdx), %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x00,0x65,0x82,0xf0,0xf7,0xff,0xff]
          vblendmps -2064(%rdx), %xmm20, %xmm24

// CHECK: vblendmps 508(%rdx){1to4}, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x10,0x65,0x42,0x7f]
          vblendmps 508(%rdx){1to4}, %xmm20, %xmm24

// CHECK: vblendmps 512(%rdx){1to4}, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x10,0x65,0x82,0x00,0x02,0x00,0x00]
          vblendmps 512(%rdx){1to4}, %xmm20, %xmm24

// CHECK: vblendmps -512(%rdx){1to4}, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x10,0x65,0x42,0x80]
          vblendmps -512(%rdx){1to4}, %xmm20, %xmm24

// CHECK: vblendmps -516(%rdx){1to4}, %xmm20, %xmm24
// CHECK:  encoding: [0x62,0x62,0x5d,0x10,0x65,0x82,0xfc,0xfd,0xff,0xff]
          vblendmps -516(%rdx){1to4}, %xmm20, %xmm24

// CHECK: vblendmps %ymm24, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0x82,0x45,0x20,0x65,0xc8]
          vblendmps %ymm24, %ymm23, %ymm17

// CHECK: vblendmps %ymm24, %ymm23, %ymm17 {%k6}
// CHECK:  encoding: [0x62,0x82,0x45,0x26,0x65,0xc8]
          vblendmps %ymm24, %ymm23, %ymm17 {%k6}

// CHECK: vblendmps %ymm24, %ymm23, %ymm17 {%k6} {z}
// CHECK:  encoding: [0x62,0x82,0x45,0xa6,0x65,0xc8]
          vblendmps %ymm24, %ymm23, %ymm17 {%k6} {z}

// CHECK: vblendmps (%rcx), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x20,0x65,0x09]
          vblendmps (%rcx), %ymm23, %ymm17

// CHECK: vblendmps 291(%rax,%r14,8), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xa2,0x45,0x20,0x65,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vblendmps 291(%rax,%r14,8), %ymm23, %ymm17

// CHECK: vblendmps (%rcx){1to8}, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x30,0x65,0x09]
          vblendmps (%rcx){1to8}, %ymm23, %ymm17

// CHECK: vblendmps 4064(%rdx), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x20,0x65,0x4a,0x7f]
          vblendmps 4064(%rdx), %ymm23, %ymm17

// CHECK: vblendmps 4096(%rdx), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x20,0x65,0x8a,0x00,0x10,0x00,0x00]
          vblendmps 4096(%rdx), %ymm23, %ymm17

// CHECK: vblendmps -4096(%rdx), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x20,0x65,0x4a,0x80]
          vblendmps -4096(%rdx), %ymm23, %ymm17

// CHECK: vblendmps -4128(%rdx), %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x20,0x65,0x8a,0xe0,0xef,0xff,0xff]
          vblendmps -4128(%rdx), %ymm23, %ymm17

// CHECK: vblendmps 508(%rdx){1to8}, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x30,0x65,0x4a,0x7f]
          vblendmps 508(%rdx){1to8}, %ymm23, %ymm17

// CHECK: vblendmps 512(%rdx){1to8}, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x30,0x65,0x8a,0x00,0x02,0x00,0x00]
          vblendmps 512(%rdx){1to8}, %ymm23, %ymm17

// CHECK: vblendmps -512(%rdx){1to8}, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x30,0x65,0x4a,0x80]
          vblendmps -512(%rdx){1to8}, %ymm23, %ymm17

// CHECK: vblendmps -516(%rdx){1to8}, %ymm23, %ymm17
// CHECK:  encoding: [0x62,0xe2,0x45,0x30,0x65,0x8a,0xfc,0xfd,0xff,0xff]
          vblendmps -516(%rdx){1to8}, %ymm23, %ymm17

// CHECK: vpblendmd %xmm26, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0x82,0x35,0x00,0x64,0xca]
          vpblendmd %xmm26, %xmm25, %xmm17

// CHECK: vpblendmd %xmm26, %xmm25, %xmm17 {%k5}
// CHECK:  encoding: [0x62,0x82,0x35,0x05,0x64,0xca]
          vpblendmd %xmm26, %xmm25, %xmm17 {%k5}

// CHECK: vpblendmd %xmm26, %xmm25, %xmm17 {%k5} {z}
// CHECK:  encoding: [0x62,0x82,0x35,0x85,0x64,0xca]
          vpblendmd %xmm26, %xmm25, %xmm17 {%k5} {z}

// CHECK: vpblendmd (%rcx), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x00,0x64,0x09]
          vpblendmd (%rcx), %xmm25, %xmm17

// CHECK: vpblendmd 291(%rax,%r14,8), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xa2,0x35,0x00,0x64,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vpblendmd 291(%rax,%r14,8), %xmm25, %xmm17

// CHECK: vpblendmd (%rcx){1to4}, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x10,0x64,0x09]
          vpblendmd (%rcx){1to4}, %xmm25, %xmm17

// CHECK: vpblendmd 2032(%rdx), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x00,0x64,0x4a,0x7f]
          vpblendmd 2032(%rdx), %xmm25, %xmm17

// CHECK: vpblendmd 2048(%rdx), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x00,0x64,0x8a,0x00,0x08,0x00,0x00]
          vpblendmd 2048(%rdx), %xmm25, %xmm17

// CHECK: vpblendmd -2048(%rdx), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x00,0x64,0x4a,0x80]
          vpblendmd -2048(%rdx), %xmm25, %xmm17

// CHECK: vpblendmd -2064(%rdx), %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x00,0x64,0x8a,0xf0,0xf7,0xff,0xff]
          vpblendmd -2064(%rdx), %xmm25, %xmm17

// CHECK: vpblendmd 508(%rdx){1to4}, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x10,0x64,0x4a,0x7f]
          vpblendmd 508(%rdx){1to4}, %xmm25, %xmm17

// CHECK: vpblendmd 512(%rdx){1to4}, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x10,0x64,0x8a,0x00,0x02,0x00,0x00]
          vpblendmd 512(%rdx){1to4}, %xmm25, %xmm17

// CHECK: vpblendmd -512(%rdx){1to4}, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x10,0x64,0x4a,0x80]
          vpblendmd -512(%rdx){1to4}, %xmm25, %xmm17

// CHECK: vpblendmd -516(%rdx){1to4}, %xmm25, %xmm17
// CHECK:  encoding: [0x62,0xe2,0x35,0x10,0x64,0x8a,0xfc,0xfd,0xff,0xff]
          vpblendmd -516(%rdx){1to4}, %xmm25, %xmm17

// CHECK: vpblendmd %ymm23, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x22,0x15,0x20,0x64,0xd7]
          vpblendmd %ymm23, %ymm29, %ymm26

// CHECK: vpblendmd %ymm23, %ymm29, %ymm26 {%k7}
// CHECK:  encoding: [0x62,0x22,0x15,0x27,0x64,0xd7]
          vpblendmd %ymm23, %ymm29, %ymm26 {%k7}

// CHECK: vpblendmd %ymm23, %ymm29, %ymm26 {%k7} {z}
// CHECK:  encoding: [0x62,0x22,0x15,0xa7,0x64,0xd7]
          vpblendmd %ymm23, %ymm29, %ymm26 {%k7} {z}

// CHECK: vpblendmd (%rcx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x20,0x64,0x11]
          vpblendmd (%rcx), %ymm29, %ymm26

// CHECK: vpblendmd 291(%rax,%r14,8), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x22,0x15,0x20,0x64,0x94,0xf0,0x23,0x01,0x00,0x00]
          vpblendmd 291(%rax,%r14,8), %ymm29, %ymm26

// CHECK: vpblendmd (%rcx){1to8}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x30,0x64,0x11]
          vpblendmd (%rcx){1to8}, %ymm29, %ymm26

// CHECK: vpblendmd 4064(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x20,0x64,0x52,0x7f]
          vpblendmd 4064(%rdx), %ymm29, %ymm26

// CHECK: vpblendmd 4096(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x20,0x64,0x92,0x00,0x10,0x00,0x00]
          vpblendmd 4096(%rdx), %ymm29, %ymm26

// CHECK: vpblendmd -4096(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x20,0x64,0x52,0x80]
          vpblendmd -4096(%rdx), %ymm29, %ymm26

// CHECK: vpblendmd -4128(%rdx), %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x20,0x64,0x92,0xe0,0xef,0xff,0xff]
          vpblendmd -4128(%rdx), %ymm29, %ymm26

// CHECK: vpblendmd 508(%rdx){1to8}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x30,0x64,0x52,0x7f]
          vpblendmd 508(%rdx){1to8}, %ymm29, %ymm26

// CHECK: vpblendmd 512(%rdx){1to8}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x30,0x64,0x92,0x00,0x02,0x00,0x00]
          vpblendmd 512(%rdx){1to8}, %ymm29, %ymm26

// CHECK: vpblendmd -512(%rdx){1to8}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x30,0x64,0x52,0x80]
          vpblendmd -512(%rdx){1to8}, %ymm29, %ymm26

// CHECK: vpblendmd -516(%rdx){1to8}, %ymm29, %ymm26
// CHECK:  encoding: [0x62,0x62,0x15,0x30,0x64,0x92,0xfc,0xfd,0xff,0xff]
          vpblendmd -516(%rdx){1to8}, %ymm29, %ymm26

// CHECK: vpblendmq %xmm17, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x22,0xa5,0x00,0x64,0xe9]
          vpblendmq %xmm17, %xmm27, %xmm29

// CHECK: vpblendmq %xmm17, %xmm27, %xmm29 {%k6}
// CHECK:  encoding: [0x62,0x22,0xa5,0x06,0x64,0xe9]
          vpblendmq %xmm17, %xmm27, %xmm29 {%k6}

// CHECK: vpblendmq %xmm17, %xmm27, %xmm29 {%k6} {z}
// CHECK:  encoding: [0x62,0x22,0xa5,0x86,0x64,0xe9]
          vpblendmq %xmm17, %xmm27, %xmm29 {%k6} {z}

// CHECK: vpblendmq (%rcx), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x00,0x64,0x29]
          vpblendmq (%rcx), %xmm27, %xmm29

// CHECK: vpblendmq 291(%rax,%r14,8), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x22,0xa5,0x00,0x64,0xac,0xf0,0x23,0x01,0x00,0x00]
          vpblendmq 291(%rax,%r14,8), %xmm27, %xmm29

// CHECK: vpblendmq (%rcx){1to2}, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x10,0x64,0x29]
          vpblendmq (%rcx){1to2}, %xmm27, %xmm29

// CHECK: vpblendmq 2032(%rdx), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x00,0x64,0x6a,0x7f]
          vpblendmq 2032(%rdx), %xmm27, %xmm29

// CHECK: vpblendmq 2048(%rdx), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x00,0x64,0xaa,0x00,0x08,0x00,0x00]
          vpblendmq 2048(%rdx), %xmm27, %xmm29

// CHECK: vpblendmq -2048(%rdx), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x00,0x64,0x6a,0x80]
          vpblendmq -2048(%rdx), %xmm27, %xmm29

// CHECK: vpblendmq -2064(%rdx), %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x00,0x64,0xaa,0xf0,0xf7,0xff,0xff]
          vpblendmq -2064(%rdx), %xmm27, %xmm29

// CHECK: vpblendmq 1016(%rdx){1to2}, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x10,0x64,0x6a,0x7f]
          vpblendmq 1016(%rdx){1to2}, %xmm27, %xmm29

// CHECK: vpblendmq 1024(%rdx){1to2}, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x10,0x64,0xaa,0x00,0x04,0x00,0x00]
          vpblendmq 1024(%rdx){1to2}, %xmm27, %xmm29

// CHECK: vpblendmq -1024(%rdx){1to2}, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x10,0x64,0x6a,0x80]
          vpblendmq -1024(%rdx){1to2}, %xmm27, %xmm29

// CHECK: vpblendmq -1032(%rdx){1to2}, %xmm27, %xmm29
// CHECK:  encoding: [0x62,0x62,0xa5,0x10,0x64,0xaa,0xf8,0xfb,0xff,0xff]
          vpblendmq -1032(%rdx){1to2}, %xmm27, %xmm29

// CHECK: vpblendmq %ymm21, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xa2,0xc5,0x20,0x64,0xed]
          vpblendmq %ymm21, %ymm23, %ymm21

// CHECK: vpblendmq %ymm21, %ymm23, %ymm21 {%k3}
// CHECK:  encoding: [0x62,0xa2,0xc5,0x23,0x64,0xed]
          vpblendmq %ymm21, %ymm23, %ymm21 {%k3}

// CHECK: vpblendmq %ymm21, %ymm23, %ymm21 {%k3} {z}
// CHECK:  encoding: [0x62,0xa2,0xc5,0xa3,0x64,0xed]
          vpblendmq %ymm21, %ymm23, %ymm21 {%k3} {z}

// CHECK: vpblendmq (%rcx), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x20,0x64,0x29]
          vpblendmq (%rcx), %ymm23, %ymm21

// CHECK: vpblendmq 291(%rax,%r14,8), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xa2,0xc5,0x20,0x64,0xac,0xf0,0x23,0x01,0x00,0x00]
          vpblendmq 291(%rax,%r14,8), %ymm23, %ymm21

// CHECK: vpblendmq (%rcx){1to4}, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x30,0x64,0x29]
          vpblendmq (%rcx){1to4}, %ymm23, %ymm21

// CHECK: vpblendmq 4064(%rdx), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x20,0x64,0x6a,0x7f]
          vpblendmq 4064(%rdx), %ymm23, %ymm21

// CHECK: vpblendmq 4096(%rdx), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x20,0x64,0xaa,0x00,0x10,0x00,0x00]
          vpblendmq 4096(%rdx), %ymm23, %ymm21

// CHECK: vpblendmq -4096(%rdx), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x20,0x64,0x6a,0x80]
          vpblendmq -4096(%rdx), %ymm23, %ymm21

// CHECK: vpblendmq -4128(%rdx), %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x20,0x64,0xaa,0xe0,0xef,0xff,0xff]
          vpblendmq -4128(%rdx), %ymm23, %ymm21

// CHECK: vpblendmq 1016(%rdx){1to4}, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x30,0x64,0x6a,0x7f]
          vpblendmq 1016(%rdx){1to4}, %ymm23, %ymm21

// CHECK: vpblendmq 1024(%rdx){1to4}, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x30,0x64,0xaa,0x00,0x04,0x00,0x00]
          vpblendmq 1024(%rdx){1to4}, %ymm23, %ymm21

// CHECK: vpblendmq -1024(%rdx){1to4}, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x30,0x64,0x6a,0x80]
          vpblendmq -1024(%rdx){1to4}, %ymm23, %ymm21

// CHECK: vpblendmq -1032(%rdx){1to4}, %ymm23, %ymm21
// CHECK:  encoding: [0x62,0xe2,0xc5,0x30,0x64,0xaa,0xf8,0xfb,0xff,0xff]
          vpblendmq -1032(%rdx){1to4}, %ymm23, %ymm21

// CHECK: vptestmd %xmm20, %xmm20, %k2
// CHECK:  encoding: [0x62,0xb2,0x5d,0x00,0x27,0xd4]
          vptestmd %xmm20, %xmm20, %k2

// CHECK: vptestmd %xmm20, %xmm20, %k2 {%k7}
// CHECK:  encoding: [0x62,0xb2,0x5d,0x07,0x27,0xd4]
          vptestmd %xmm20, %xmm20, %k2 {%k7}

// CHECK: vptestmd (%rcx), %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x00,0x27,0x11]
          vptestmd (%rcx), %xmm20, %k2

// CHECK: vptestmd 291(%rax,%r14,8), %xmm20, %k2
// CHECK:  encoding: [0x62,0xb2,0x5d,0x00,0x27,0x94,0xf0,0x23,0x01,0x00,0x00]
          vptestmd 291(%rax,%r14,8), %xmm20, %k2

// CHECK: vptestmd (%rcx){1to4}, %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x10,0x27,0x11]
          vptestmd (%rcx){1to4}, %xmm20, %k2

// CHECK: vptestmd 2032(%rdx), %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x00,0x27,0x52,0x7f]
          vptestmd 2032(%rdx), %xmm20, %k2

// CHECK: vptestmd 2048(%rdx), %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x00,0x27,0x92,0x00,0x08,0x00,0x00]
          vptestmd 2048(%rdx), %xmm20, %k2

// CHECK: vptestmd -2048(%rdx), %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x00,0x27,0x52,0x80]
          vptestmd -2048(%rdx), %xmm20, %k2

// CHECK: vptestmd -2064(%rdx), %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x00,0x27,0x92,0xf0,0xf7,0xff,0xff]
          vptestmd -2064(%rdx), %xmm20, %k2

// CHECK: vptestmd 508(%rdx){1to4}, %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x10,0x27,0x52,0x7f]
          vptestmd 508(%rdx){1to4}, %xmm20, %k2

// CHECK: vptestmd 512(%rdx){1to4}, %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x10,0x27,0x92,0x00,0x02,0x00,0x00]
          vptestmd 512(%rdx){1to4}, %xmm20, %k2

// CHECK: vptestmd -512(%rdx){1to4}, %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x10,0x27,0x52,0x80]
          vptestmd -512(%rdx){1to4}, %xmm20, %k2

// CHECK: vptestmd -516(%rdx){1to4}, %xmm20, %k2
// CHECK:  encoding: [0x62,0xf2,0x5d,0x10,0x27,0x92,0xfc,0xfd,0xff,0xff]
          vptestmd -516(%rdx){1to4}, %xmm20, %k2

// CHECK: vptestmd %ymm17, %ymm20, %k3
// CHECK:  encoding: [0x62,0xb2,0x5d,0x20,0x27,0xd9]
          vptestmd %ymm17, %ymm20, %k3

// CHECK: vptestmd %ymm17, %ymm20, %k3 {%k5}
// CHECK:  encoding: [0x62,0xb2,0x5d,0x25,0x27,0xd9]
          vptestmd %ymm17, %ymm20, %k3 {%k5}

// CHECK: vptestmd (%rcx), %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x20,0x27,0x19]
          vptestmd (%rcx), %ymm20, %k3

// CHECK: vptestmd 291(%rax,%r14,8), %ymm20, %k3
// CHECK:  encoding: [0x62,0xb2,0x5d,0x20,0x27,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vptestmd 291(%rax,%r14,8), %ymm20, %k3

// CHECK: vptestmd (%rcx){1to8}, %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x30,0x27,0x19]
          vptestmd (%rcx){1to8}, %ymm20, %k3

// CHECK: vptestmd 4064(%rdx), %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x20,0x27,0x5a,0x7f]
          vptestmd 4064(%rdx), %ymm20, %k3

// CHECK: vptestmd 4096(%rdx), %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x20,0x27,0x9a,0x00,0x10,0x00,0x00]
          vptestmd 4096(%rdx), %ymm20, %k3

// CHECK: vptestmd -4096(%rdx), %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x20,0x27,0x5a,0x80]
          vptestmd -4096(%rdx), %ymm20, %k3

// CHECK: vptestmd -4128(%rdx), %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x20,0x27,0x9a,0xe0,0xef,0xff,0xff]
          vptestmd -4128(%rdx), %ymm20, %k3

// CHECK: vptestmd 508(%rdx){1to8}, %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x30,0x27,0x5a,0x7f]
          vptestmd 508(%rdx){1to8}, %ymm20, %k3

// CHECK: vptestmd 512(%rdx){1to8}, %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x30,0x27,0x9a,0x00,0x02,0x00,0x00]
          vptestmd 512(%rdx){1to8}, %ymm20, %k3

// CHECK: vptestmd -512(%rdx){1to8}, %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x30,0x27,0x5a,0x80]
          vptestmd -512(%rdx){1to8}, %ymm20, %k3

// CHECK: vptestmd -516(%rdx){1to8}, %ymm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5d,0x30,0x27,0x9a,0xfc,0xfd,0xff,0xff]
          vptestmd -516(%rdx){1to8}, %ymm20, %k3

// CHECK: vptestmq %xmm28, %xmm22, %k4
// CHECK:  encoding: [0x62,0x92,0xcd,0x00,0x27,0xe4]
          vptestmq %xmm28, %xmm22, %k4

// CHECK: vptestmq %xmm28, %xmm22, %k4 {%k3}
// CHECK:  encoding: [0x62,0x92,0xcd,0x03,0x27,0xe4]
          vptestmq %xmm28, %xmm22, %k4 {%k3}

// CHECK: vptestmq (%rcx), %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x00,0x27,0x21]
          vptestmq (%rcx), %xmm22, %k4

// CHECK: vptestmq 291(%rax,%r14,8), %xmm22, %k4
// CHECK:  encoding: [0x62,0xb2,0xcd,0x00,0x27,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vptestmq 291(%rax,%r14,8), %xmm22, %k4

// CHECK: vptestmq (%rcx){1to2}, %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x10,0x27,0x21]
          vptestmq (%rcx){1to2}, %xmm22, %k4

// CHECK: vptestmq 2032(%rdx), %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x00,0x27,0x62,0x7f]
          vptestmq 2032(%rdx), %xmm22, %k4

// CHECK: vptestmq 2048(%rdx), %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x00,0x27,0xa2,0x00,0x08,0x00,0x00]
          vptestmq 2048(%rdx), %xmm22, %k4

// CHECK: vptestmq -2048(%rdx), %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x00,0x27,0x62,0x80]
          vptestmq -2048(%rdx), %xmm22, %k4

// CHECK: vptestmq -2064(%rdx), %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x00,0x27,0xa2,0xf0,0xf7,0xff,0xff]
          vptestmq -2064(%rdx), %xmm22, %k4

// CHECK: vptestmq 1016(%rdx){1to2}, %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x10,0x27,0x62,0x7f]
          vptestmq 1016(%rdx){1to2}, %xmm22, %k4

// CHECK: vptestmq 1024(%rdx){1to2}, %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x10,0x27,0xa2,0x00,0x04,0x00,0x00]
          vptestmq 1024(%rdx){1to2}, %xmm22, %k4

// CHECK: vptestmq -1024(%rdx){1to2}, %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x10,0x27,0x62,0x80]
          vptestmq -1024(%rdx){1to2}, %xmm22, %k4

// CHECK: vptestmq -1032(%rdx){1to2}, %xmm22, %k4
// CHECK:  encoding: [0x62,0xf2,0xcd,0x10,0x27,0xa2,0xf8,0xfb,0xff,0xff]
          vptestmq -1032(%rdx){1to2}, %xmm22, %k4

// CHECK: vptestmq %ymm20, %ymm21, %k3
// CHECK:  encoding: [0x62,0xb2,0xd5,0x20,0x27,0xdc]
          vptestmq %ymm20, %ymm21, %k3

// CHECK: vptestmq %ymm20, %ymm21, %k3 {%k7}
// CHECK:  encoding: [0x62,0xb2,0xd5,0x27,0x27,0xdc]
          vptestmq %ymm20, %ymm21, %k3 {%k7}

// CHECK: vptestmq (%rcx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x20,0x27,0x19]
          vptestmq (%rcx), %ymm21, %k3

// CHECK: vptestmq 291(%rax,%r14,8), %ymm21, %k3
// CHECK:  encoding: [0x62,0xb2,0xd5,0x20,0x27,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vptestmq 291(%rax,%r14,8), %ymm21, %k3

// CHECK: vptestmq (%rcx){1to4}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x30,0x27,0x19]
          vptestmq (%rcx){1to4}, %ymm21, %k3

// CHECK: vptestmq 4064(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x20,0x27,0x5a,0x7f]
          vptestmq 4064(%rdx), %ymm21, %k3

// CHECK: vptestmq 4096(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x20,0x27,0x9a,0x00,0x10,0x00,0x00]
          vptestmq 4096(%rdx), %ymm21, %k3

// CHECK: vptestmq -4096(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x20,0x27,0x5a,0x80]
          vptestmq -4096(%rdx), %ymm21, %k3

// CHECK: vptestmq -4128(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x20,0x27,0x9a,0xe0,0xef,0xff,0xff]
          vptestmq -4128(%rdx), %ymm21, %k3

// CHECK: vptestmq 1016(%rdx){1to4}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x30,0x27,0x5a,0x7f]
          vptestmq 1016(%rdx){1to4}, %ymm21, %k3

// CHECK: vptestmq 1024(%rdx){1to4}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x30,0x27,0x9a,0x00,0x04,0x00,0x00]
          vptestmq 1024(%rdx){1to4}, %ymm21, %k3

// CHECK: vptestmq -1024(%rdx){1to4}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0xd5,0x30,0x27,0x5a,0x80]
          vptestmq -1024(%rdx){1to4}, %ymm21, %k3

// CHECK: vptestnmd %xmm22, %xmm20, %k3
// CHECK:  encoding: [0x62,0xb2,0x5e,0x00,0x27,0xde]
          vptestnmd %xmm22, %xmm20, %k3

// CHECK: vptestnmd %xmm22, %xmm20, %k3 {%k7}
// CHECK:  encoding: [0x62,0xb2,0x5e,0x07,0x27,0xde]
          vptestnmd %xmm22, %xmm20, %k3 {%k7}

// CHECK: vptestnmd (%rcx), %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x00,0x27,0x19]
          vptestnmd (%rcx), %xmm20, %k3

// CHECK: vptestnmd 291(%rax,%r14,8), %xmm20, %k3
// CHECK:  encoding: [0x62,0xb2,0x5e,0x00,0x27,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vptestnmd 291(%rax,%r14,8), %xmm20, %k3

// CHECK: vptestnmd (%rcx){1to4}, %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x10,0x27,0x19]
          vptestnmd (%rcx){1to4}, %xmm20, %k3

// CHECK: vptestnmd 2032(%rdx), %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x00,0x27,0x5a,0x7f]
          vptestnmd 2032(%rdx), %xmm20, %k3

// CHECK: vptestnmd 2048(%rdx), %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x00,0x27,0x9a,0x00,0x08,0x00,0x00]
          vptestnmd 2048(%rdx), %xmm20, %k3

// CHECK: vptestnmd -2048(%rdx), %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x00,0x27,0x5a,0x80]
          vptestnmd -2048(%rdx), %xmm20, %k3

// CHECK: vptestnmd -2064(%rdx), %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x00,0x27,0x9a,0xf0,0xf7,0xff,0xff]
          vptestnmd -2064(%rdx), %xmm20, %k3

// CHECK: vptestnmd 508(%rdx){1to4}, %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x10,0x27,0x5a,0x7f]
          vptestnmd 508(%rdx){1to4}, %xmm20, %k3

// CHECK: vptestnmd 512(%rdx){1to4}, %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x10,0x27,0x9a,0x00,0x02,0x00,0x00]
          vptestnmd 512(%rdx){1to4}, %xmm20, %k3

// CHECK: vptestnmd -512(%rdx){1to4}, %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x10,0x27,0x5a,0x80]
          vptestnmd -512(%rdx){1to4}, %xmm20, %k3

// CHECK: vptestnmd -516(%rdx){1to4}, %xmm20, %k3
// CHECK:  encoding: [0x62,0xf2,0x5e,0x10,0x27,0x9a,0xfc,0xfd,0xff,0xff]
          vptestnmd -516(%rdx){1to4}, %xmm20, %k3

// CHECK: vptestnmd %ymm23, %ymm21, %k3
// CHECK:  encoding: [0x62,0xb2,0x56,0x20,0x27,0xdf]
          vptestnmd %ymm23, %ymm21, %k3

// CHECK: vptestnmd %ymm23, %ymm21, %k3 {%k7}
// CHECK:  encoding: [0x62,0xb2,0x56,0x27,0x27,0xdf]
          vptestnmd %ymm23, %ymm21, %k3 {%k7}

// CHECK: vptestnmd (%rcx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x20,0x27,0x19]
          vptestnmd (%rcx), %ymm21, %k3

// CHECK: vptestnmd 291(%rax,%r14,8), %ymm21, %k3
// CHECK:  encoding: [0x62,0xb2,0x56,0x20,0x27,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vptestnmd 291(%rax,%r14,8), %ymm21, %k3

// CHECK: vptestnmd (%rcx){1to8}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x30,0x27,0x19]
          vptestnmd (%rcx){1to8}, %ymm21, %k3

// CHECK: vptestnmd 4064(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x20,0x27,0x5a,0x7f]
          vptestnmd 4064(%rdx), %ymm21, %k3

// CHECK: vptestnmd 4096(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x20,0x27,0x9a,0x00,0x10,0x00,0x00]
          vptestnmd 4096(%rdx), %ymm21, %k3

// CHECK: vptestnmd -4096(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x20,0x27,0x5a,0x80]
          vptestnmd -4096(%rdx), %ymm21, %k3

// CHECK: vptestnmd -4128(%rdx), %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x20,0x27,0x9a,0xe0,0xef,0xff,0xff]
          vptestnmd -4128(%rdx), %ymm21, %k3

// CHECK: vptestnmd 508(%rdx){1to8}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x30,0x27,0x5a,0x7f]
          vptestnmd 508(%rdx){1to8}, %ymm21, %k3

// CHECK: vptestnmd 512(%rdx){1to8}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x30,0x27,0x9a,0x00,0x02,0x00,0x00]
          vptestnmd 512(%rdx){1to8}, %ymm21, %k3

// CHECK: vptestnmd -512(%rdx){1to8}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x30,0x27,0x5a,0x80]
          vptestnmd -512(%rdx){1to8}, %ymm21, %k3

// CHECK: vptestnmd -516(%rdx){1to8}, %ymm21, %k3
// CHECK:  encoding: [0x62,0xf2,0x56,0x30,0x27,0x9a,0xfc,0xfd,0xff,0xff]
          vptestnmd -516(%rdx){1to8}, %ymm21, %k3

// CHECK: vptestnmq %xmm21, %xmm20, %k5
// CHECK:  encoding: [0x62,0xb2,0xde,0x00,0x27,0xed]
          vptestnmq %xmm21, %xmm20, %k5

// CHECK: vptestnmq %xmm21, %xmm20, %k5 {%k5}
// CHECK:  encoding: [0x62,0xb2,0xde,0x05,0x27,0xed]
          vptestnmq %xmm21, %xmm20, %k5 {%k5}

// CHECK: vptestnmq (%rcx), %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x00,0x27,0x29]
          vptestnmq (%rcx), %xmm20, %k5

// CHECK: vptestnmq 291(%rax,%r14,8), %xmm20, %k5
// CHECK:  encoding: [0x62,0xb2,0xde,0x00,0x27,0xac,0xf0,0x23,0x01,0x00,0x00]
          vptestnmq 291(%rax,%r14,8), %xmm20, %k5

// CHECK: vptestnmq (%rcx){1to2}, %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x10,0x27,0x29]
          vptestnmq (%rcx){1to2}, %xmm20, %k5

// CHECK: vptestnmq 2032(%rdx), %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x00,0x27,0x6a,0x7f]
          vptestnmq 2032(%rdx), %xmm20, %k5

// CHECK: vptestnmq 2048(%rdx), %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x00,0x27,0xaa,0x00,0x08,0x00,0x00]
          vptestnmq 2048(%rdx), %xmm20, %k5

// CHECK: vptestnmq -2048(%rdx), %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x00,0x27,0x6a,0x80]
          vptestnmq -2048(%rdx), %xmm20, %k5

// CHECK: vptestnmq -2064(%rdx), %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x00,0x27,0xaa,0xf0,0xf7,0xff,0xff]
          vptestnmq -2064(%rdx), %xmm20, %k5

// CHECK: vptestnmq 1016(%rdx){1to2}, %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x10,0x27,0x6a,0x7f]
          vptestnmq 1016(%rdx){1to2}, %xmm20, %k5

// CHECK: vptestnmq 1024(%rdx){1to2}, %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x10,0x27,0xaa,0x00,0x04,0x00,0x00]
          vptestnmq 1024(%rdx){1to2}, %xmm20, %k5

// CHECK: vptestnmq -1024(%rdx){1to2}, %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x10,0x27,0x6a,0x80]
          vptestnmq -1024(%rdx){1to2}, %xmm20, %k5

// CHECK: vptestnmq -1032(%rdx){1to2}, %xmm20, %k5
// CHECK:  encoding: [0x62,0xf2,0xde,0x10,0x27,0xaa,0xf8,0xfb,0xff,0xff]
          vptestnmq -1032(%rdx){1to2}, %xmm20, %k5

// CHECK: vptestnmq %ymm21, %ymm24, %k4
// CHECK:  encoding: [0x62,0xb2,0xbe,0x20,0x27,0xe5]
          vptestnmq %ymm21, %ymm24, %k4

// CHECK: vptestnmq %ymm21, %ymm24, %k4 {%k3}
// CHECK:  encoding: [0x62,0xb2,0xbe,0x23,0x27,0xe5]
          vptestnmq %ymm21, %ymm24, %k4 {%k3}

// CHECK: vptestnmq (%rcx), %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x20,0x27,0x21]
          vptestnmq (%rcx), %ymm24, %k4

// CHECK: vptestnmq 291(%rax,%r14,8), %ymm24, %k4
// CHECK:  encoding: [0x62,0xb2,0xbe,0x20,0x27,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vptestnmq 291(%rax,%r14,8), %ymm24, %k4

// CHECK: vptestnmq (%rcx){1to4}, %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x30,0x27,0x21]
          vptestnmq (%rcx){1to4}, %ymm24, %k4

// CHECK: vptestnmq 4064(%rdx), %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x20,0x27,0x62,0x7f]
          vptestnmq 4064(%rdx), %ymm24, %k4

// CHECK: vptestnmq 4096(%rdx), %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x20,0x27,0xa2,0x00,0x10,0x00,0x00]
          vptestnmq 4096(%rdx), %ymm24, %k4

// CHECK: vptestnmq -4096(%rdx), %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x20,0x27,0x62,0x80]
          vptestnmq -4096(%rdx), %ymm24, %k4

// CHECK: vptestnmq -4128(%rdx), %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x20,0x27,0xa2,0xe0,0xef,0xff,0xff]
          vptestnmq -4128(%rdx), %ymm24, %k4

// CHECK: vptestnmq 1016(%rdx){1to4}, %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x30,0x27,0x62,0x7f]
          vptestnmq 1016(%rdx){1to4}, %ymm24, %k4

// CHECK: vptestnmq 1024(%rdx){1to4}, %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x30,0x27,0xa2,0x00,0x04,0x00,0x00]
          vptestnmq 1024(%rdx){1to4}, %ymm24, %k4

// CHECK: vptestnmq -1024(%rdx){1to4}, %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x30,0x27,0x62,0x80]
          vptestnmq -1024(%rdx){1to4}, %ymm24, %k4

// CHECK: vptestnmq -1032(%rdx){1to4}, %ymm24, %k4
// CHECK:  encoding: [0x62,0xf2,0xbe,0x30,0x27,0xa2,0xf8,0xfb,0xff,0xff]
          vptestnmq -1032(%rdx){1to4}, %ymm24, %k4

// CHECK: vpmovd2m %xmm27, %k3
// CHECK:  encoding: [0x62,0x92,0x7e,0x08,0x39,0xdb]
          vpmovd2m %xmm27, %k3

// CHECK: vpmovd2m %ymm28, %k4
// CHECK:  encoding: [0x62,0x92,0x7e,0x28,0x39,0xe4]
          vpmovd2m %ymm28, %k4

// CHECK: vpmovq2m %xmm28, %k5
// CHECK:  encoding: [0x62,0x92,0xfe,0x08,0x39,0xec]
          vpmovq2m %xmm28, %k5

// CHECK: vpmovq2m %ymm29, %k4
// CHECK:  encoding: [0x62,0x92,0xfe,0x28,0x39,0xe5]
          vpmovq2m %ymm29, %k4

// CHECK: vpmovm2d %k2, %xmm29
// CHECK:  encoding: [0x62,0x62,0x7e,0x08,0x38,0xea]
          vpmovm2d %k2, %xmm29

// CHECK: vpmovm2d %k5, %ymm20
// CHECK:  encoding: [0x62,0xe2,0x7e,0x28,0x38,0xe5]
          vpmovm2d %k5, %ymm20

// CHECK: vpmovm2q %k5, %xmm17
// CHECK:  encoding: [0x62,0xe2,0xfe,0x08,0x38,0xcd]
          vpmovm2q %k5, %xmm17

// CHECK: vpmovm2q %k2, %ymm30
// CHECK:  encoding: [0x62,0x62,0xfe,0x28,0x38,0xf2]
          vpmovm2q %k2, %ymm30
