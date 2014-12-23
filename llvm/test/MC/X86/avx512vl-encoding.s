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
