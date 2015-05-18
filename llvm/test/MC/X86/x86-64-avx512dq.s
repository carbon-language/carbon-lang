// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512dq  --show-encoding %s | FileCheck %s

// CHECK: vpmullq %zmm18, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xa2,0xbd,0x40,0x40,0xd2]
          vpmullq %zmm18, %zmm24, %zmm18

// CHECK: vpmullq %zmm18, %zmm24, %zmm18 {%k2}
// CHECK:  encoding: [0x62,0xa2,0xbd,0x42,0x40,0xd2]
          vpmullq %zmm18, %zmm24, %zmm18 {%k2}

// CHECK: vpmullq %zmm18, %zmm24, %zmm18 {%k2} {z}
// CHECK:  encoding: [0x62,0xa2,0xbd,0xc2,0x40,0xd2]
          vpmullq %zmm18, %zmm24, %zmm18 {%k2} {z}

// CHECK: vpmullq (%rcx), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x40,0x40,0x11]
          vpmullq (%rcx), %zmm24, %zmm18

// CHECK: vpmullq 291(%rax,%r14,8), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xa2,0xbd,0x40,0x40,0x94,0xf0,0x23,0x01,0x00,0x00]
          vpmullq 291(%rax,%r14,8), %zmm24, %zmm18

// CHECK: vpmullq (%rcx){1to8}, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x50,0x40,0x11]
          vpmullq (%rcx){1to8}, %zmm24, %zmm18

// CHECK: vpmullq 8128(%rdx), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x40,0x40,0x52,0x7f]
          vpmullq 8128(%rdx), %zmm24, %zmm18

// CHECK: vpmullq 8192(%rdx), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x40,0x40,0x92,0x00,0x20,0x00,0x00]
          vpmullq 8192(%rdx), %zmm24, %zmm18

// CHECK: vpmullq -8192(%rdx), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x40,0x40,0x52,0x80]
          vpmullq -8192(%rdx), %zmm24, %zmm18

// CHECK: vpmullq -8256(%rdx), %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x40,0x40,0x92,0xc0,0xdf,0xff,0xff]
          vpmullq -8256(%rdx), %zmm24, %zmm18

// CHECK: vpmullq 1016(%rdx){1to8}, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x50,0x40,0x52,0x7f]
          vpmullq 1016(%rdx){1to8}, %zmm24, %zmm18

// CHECK: vpmullq 1024(%rdx){1to8}, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x50,0x40,0x92,0x00,0x04,0x00,0x00]
          vpmullq 1024(%rdx){1to8}, %zmm24, %zmm18

// CHECK: vpmullq -1024(%rdx){1to8}, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x50,0x40,0x52,0x80]
          vpmullq -1024(%rdx){1to8}, %zmm24, %zmm18

// CHECK: vpmullq -1032(%rdx){1to8}, %zmm24, %zmm18
// CHECK:  encoding: [0x62,0xe2,0xbd,0x50,0x40,0x92,0xf8,0xfb,0xff,0xff]
          vpmullq -1032(%rdx){1to8}, %zmm24, %zmm18

// CHECK: kandb  %k6, %k5, %k2
// CHECK:  encoding: [0xc5,0xd5,0x41,0xd6]
          kandb  %k6, %k5, %k2

// CHECK: kandnb %k4, %k6, %k5
// CHECK:  encoding: [0xc5,0xcd,0x42,0xec]
          kandnb %k4, %k6, %k5

// CHECK: korb   %k5, %k4, %k4
// CHECK:  encoding: [0xc5,0xdd,0x45,0xe5]
          korb   %k5, %k4, %k4

// CHECK: kxnorb %k7, %k6, %k4
// CHECK:  encoding: [0xc5,0xcd,0x46,0xe7]
          kxnorb %k7, %k6, %k4

// CHECK: kxorb  %k5, %k6, %k4
// CHECK:  encoding: [0xc5,0xcd,0x47,0xe5]
          kxorb  %k5, %k6, %k4

// CHECK: knotb  %k4, %k5
// CHECK:  encoding: [0xc5,0xf9,0x44,0xec]
          knotb  %k4, %k5

// CHECK: knotb  %k3, %k3
// CHECK:  encoding: [0xc5,0xf9,0x44,0xdb]
          knotb  %k3, %k3

// CHECK: kmovb  %k3, %k5
// CHECK:  encoding: [0xc5,0xf9,0x90,0xeb]
          kmovb  %k3, %k5

// CHECK: kmovb  (%rcx), %k5
// CHECK:  encoding: [0xc5,0xf9,0x90,0x29]
          kmovb  (%rcx), %k5

// CHECK: kmovb  4660(%rax,%r14,8), %k5
// CHECK:  encoding: [0xc4,0xa1,0x79,0x90,0xac,0xf0,0x34,0x12,0x00,0x00]
          kmovb  4660(%rax,%r14,8), %k5

// CHECK: kmovb  %k2, (%rcx)
// CHECK:  encoding: [0xc5,0xf9,0x91,0x11]
          kmovb  %k2, (%rcx)

// CHECK: kmovb  %k2, 4660(%rax,%r14,8)
// CHECK:  encoding: [0xc4,0xa1,0x79,0x91,0x94,0xf0,0x34,0x12,0x00,0x00]
          kmovb  %k2, 4660(%rax,%r14,8)

// CHECK: kmovb  %eax, %k2
// CHECK:  encoding: [0xc5,0xf9,0x92,0xd0]
          kmovb  %eax, %k2

// CHECK: kmovb  %ebp, %k2
// CHECK:  encoding: [0xc5,0xf9,0x92,0xd5]
          kmovb  %ebp, %k2

// CHECK: kmovb  %r13d, %k2
// CHECK:  encoding: [0xc4,0xc1,0x79,0x92,0xd5]
          kmovb  %r13d, %k2

// CHECK: kmovb  %k3, %eax
// CHECK:  encoding: [0xc5,0xf9,0x93,0xc3]
          kmovb  %k3, %eax

// CHECK: kmovb  %k3, %ebp
// CHECK:  encoding: [0xc5,0xf9,0x93,0xeb]
          kmovb  %k3, %ebp

// CHECK: kmovb  %k3, %r13d
// CHECK:  encoding: [0xc5,0x79,0x93,0xeb]
          kmovb  %k3, %r13d

// CHECK: vandpd %zmm27, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0x81,0x9d,0x40,0x54,0xdb]
          vandpd %zmm27, %zmm28, %zmm19

// CHECK: vandpd %zmm27, %zmm28, %zmm19 {%k5}
// CHECK:  encoding: [0x62,0x81,0x9d,0x45,0x54,0xdb]
          vandpd %zmm27, %zmm28, %zmm19 {%k5}

// CHECK: vandpd %zmm27, %zmm28, %zmm19 {%k5} {z}
// CHECK:  encoding: [0x62,0x81,0x9d,0xc5,0x54,0xdb]
          vandpd %zmm27, %zmm28, %zmm19 {%k5} {z}

// CHECK: vandpd (%rcx), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x54,0x19]
          vandpd (%rcx), %zmm28, %zmm19

// CHECK: vandpd 291(%rax,%r14,8), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xa1,0x9d,0x40,0x54,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vandpd 291(%rax,%r14,8), %zmm28, %zmm19

// CHECK: vandpd (%rcx){1to8}, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x54,0x19]
          vandpd (%rcx){1to8}, %zmm28, %zmm19

// CHECK: vandpd 8128(%rdx), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x54,0x5a,0x7f]
          vandpd 8128(%rdx), %zmm28, %zmm19

// CHECK: vandpd 8192(%rdx), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x54,0x9a,0x00,0x20,0x00,0x00]
          vandpd 8192(%rdx), %zmm28, %zmm19

// CHECK: vandpd -8192(%rdx), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x54,0x5a,0x80]
          vandpd -8192(%rdx), %zmm28, %zmm19

// CHECK: vandpd -8256(%rdx), %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x54,0x9a,0xc0,0xdf,0xff,0xff]
          vandpd -8256(%rdx), %zmm28, %zmm19

// CHECK: vandpd 1016(%rdx){1to8}, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x54,0x5a,0x7f]
          vandpd 1016(%rdx){1to8}, %zmm28, %zmm19

// CHECK: vandpd 1024(%rdx){1to8}, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x54,0x9a,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to8}, %zmm28, %zmm19

// CHECK: vandpd -1024(%rdx){1to8}, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x54,0x5a,0x80]
          vandpd -1024(%rdx){1to8}, %zmm28, %zmm19

// CHECK: vandpd -1032(%rdx){1to8}, %zmm28, %zmm19
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x54,0x9a,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to8}, %zmm28, %zmm19

// CHECK: vandps %zmm25, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0x81,0x4c,0x40,0x54,0xc9]
          vandps %zmm25, %zmm22, %zmm17

// CHECK: vandps %zmm25, %zmm22, %zmm17 {%k4}
// CHECK:  encoding: [0x62,0x81,0x4c,0x44,0x54,0xc9]
          vandps %zmm25, %zmm22, %zmm17 {%k4}

// CHECK: vandps %zmm25, %zmm22, %zmm17 {%k4} {z}
// CHECK:  encoding: [0x62,0x81,0x4c,0xc4,0x54,0xc9]
          vandps %zmm25, %zmm22, %zmm17 {%k4} {z}

// CHECK: vandps (%rcx), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x40,0x54,0x09]
          vandps (%rcx), %zmm22, %zmm17

// CHECK: vandps 291(%rax,%r14,8), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xa1,0x4c,0x40,0x54,0x8c,0xf0,0x23,0x01,0x00,0x00]
          vandps 291(%rax,%r14,8), %zmm22, %zmm17

// CHECK: vandps (%rcx){1to16}, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x50,0x54,0x09]
          vandps (%rcx){1to16}, %zmm22, %zmm17

// CHECK: vandps 8128(%rdx), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x40,0x54,0x4a,0x7f]
          vandps 8128(%rdx), %zmm22, %zmm17

// CHECK: vandps 8192(%rdx), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x40,0x54,0x8a,0x00,0x20,0x00,0x00]
          vandps 8192(%rdx), %zmm22, %zmm17

// CHECK: vandps -8192(%rdx), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x40,0x54,0x4a,0x80]
          vandps -8192(%rdx), %zmm22, %zmm17

// CHECK: vandps -8256(%rdx), %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x40,0x54,0x8a,0xc0,0xdf,0xff,0xff]
          vandps -8256(%rdx), %zmm22, %zmm17

// CHECK: vandps 508(%rdx){1to16}, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x50,0x54,0x4a,0x7f]
          vandps 508(%rdx){1to16}, %zmm22, %zmm17

// CHECK: vandps 512(%rdx){1to16}, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x50,0x54,0x8a,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to16}, %zmm22, %zmm17

// CHECK: vandps -512(%rdx){1to16}, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x50,0x54,0x4a,0x80]
          vandps -512(%rdx){1to16}, %zmm22, %zmm17

// CHECK: vandps -516(%rdx){1to16}, %zmm22, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x4c,0x50,0x54,0x8a,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to16}, %zmm22, %zmm17

// CHECK: vandnpd %zmm22, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xa1,0xf5,0x40,0x55,0xe6]
          vandnpd %zmm22, %zmm17, %zmm20

// CHECK: vandnpd %zmm22, %zmm17, %zmm20 {%k1}
// CHECK:  encoding: [0x62,0xa1,0xf5,0x41,0x55,0xe6]
          vandnpd %zmm22, %zmm17, %zmm20 {%k1}

// CHECK: vandnpd %zmm22, %zmm17, %zmm20 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0xf5,0xc1,0x55,0xe6]
          vandnpd %zmm22, %zmm17, %zmm20 {%k1} {z}

// CHECK: vandnpd (%rcx), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x40,0x55,0x21]
          vandnpd (%rcx), %zmm17, %zmm20

// CHECK: vandnpd 291(%rax,%r14,8), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xa1,0xf5,0x40,0x55,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vandnpd 291(%rax,%r14,8), %zmm17, %zmm20

// CHECK: vandnpd (%rcx){1to8}, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x50,0x55,0x21]
          vandnpd (%rcx){1to8}, %zmm17, %zmm20

// CHECK: vandnpd 8128(%rdx), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x40,0x55,0x62,0x7f]
          vandnpd 8128(%rdx), %zmm17, %zmm20

// CHECK: vandnpd 8192(%rdx), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x40,0x55,0xa2,0x00,0x20,0x00,0x00]
          vandnpd 8192(%rdx), %zmm17, %zmm20

// CHECK: vandnpd -8192(%rdx), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x40,0x55,0x62,0x80]
          vandnpd -8192(%rdx), %zmm17, %zmm20

// CHECK: vandnpd -8256(%rdx), %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x40,0x55,0xa2,0xc0,0xdf,0xff,0xff]
          vandnpd -8256(%rdx), %zmm17, %zmm20

// CHECK: vandnpd 1016(%rdx){1to8}, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x50,0x55,0x62,0x7f]
          vandnpd 1016(%rdx){1to8}, %zmm17, %zmm20

// CHECK: vandnpd 1024(%rdx){1to8}, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x50,0x55,0xa2,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to8}, %zmm17, %zmm20

// CHECK: vandnpd -1024(%rdx){1to8}, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x50,0x55,0x62,0x80]
          vandnpd -1024(%rdx){1to8}, %zmm17, %zmm20

// CHECK: vandnpd -1032(%rdx){1to8}, %zmm17, %zmm20
// CHECK:  encoding: [0x62,0xe1,0xf5,0x50,0x55,0xa2,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to8}, %zmm17, %zmm20

// CHECK: vandnps %zmm19, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xa1,0x74,0x40,0x55,0xf3]
          vandnps %zmm19, %zmm17, %zmm22

// CHECK: vandnps %zmm19, %zmm17, %zmm22 {%k2}
// CHECK:  encoding: [0x62,0xa1,0x74,0x42,0x55,0xf3]
          vandnps %zmm19, %zmm17, %zmm22 {%k2}

// CHECK: vandnps %zmm19, %zmm17, %zmm22 {%k2} {z}
// CHECK:  encoding: [0x62,0xa1,0x74,0xc2,0x55,0xf3]
          vandnps %zmm19, %zmm17, %zmm22 {%k2} {z}

// CHECK: vandnps (%rcx), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x40,0x55,0x31]
          vandnps (%rcx), %zmm17, %zmm22

// CHECK: vandnps 291(%rax,%r14,8), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xa1,0x74,0x40,0x55,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vandnps 291(%rax,%r14,8), %zmm17, %zmm22

// CHECK: vandnps (%rcx){1to16}, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x50,0x55,0x31]
          vandnps (%rcx){1to16}, %zmm17, %zmm22

// CHECK: vandnps 8128(%rdx), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x40,0x55,0x72,0x7f]
          vandnps 8128(%rdx), %zmm17, %zmm22

// CHECK: vandnps 8192(%rdx), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x40,0x55,0xb2,0x00,0x20,0x00,0x00]
          vandnps 8192(%rdx), %zmm17, %zmm22

// CHECK: vandnps -8192(%rdx), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x40,0x55,0x72,0x80]
          vandnps -8192(%rdx), %zmm17, %zmm22

// CHECK: vandnps -8256(%rdx), %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x40,0x55,0xb2,0xc0,0xdf,0xff,0xff]
          vandnps -8256(%rdx), %zmm17, %zmm22

// CHECK: vandnps 508(%rdx){1to16}, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x50,0x55,0x72,0x7f]
          vandnps 508(%rdx){1to16}, %zmm17, %zmm22

// CHECK: vandnps 512(%rdx){1to16}, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x50,0x55,0xb2,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to16}, %zmm17, %zmm22

// CHECK: vandnps -512(%rdx){1to16}, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x50,0x55,0x72,0x80]
          vandnps -512(%rdx){1to16}, %zmm17, %zmm22

// CHECK: vandnps -516(%rdx){1to16}, %zmm17, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x74,0x50,0x55,0xb2,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to16}, %zmm17, %zmm22

// CHECK: vorpd  %zmm21, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x21,0xcd,0x40,0x56,0xf5]
          vorpd  %zmm21, %zmm22, %zmm30

// CHECK: vorpd  %zmm21, %zmm22, %zmm30 {%k6}
// CHECK:  encoding: [0x62,0x21,0xcd,0x46,0x56,0xf5]
          vorpd  %zmm21, %zmm22, %zmm30 {%k6}

// CHECK: vorpd  %zmm21, %zmm22, %zmm30 {%k6} {z}
// CHECK:  encoding: [0x62,0x21,0xcd,0xc6,0x56,0xf5]
          vorpd  %zmm21, %zmm22, %zmm30 {%k6} {z}

// CHECK: vorpd  (%rcx), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x56,0x31]
          vorpd  (%rcx), %zmm22, %zmm30

// CHECK: vorpd  291(%rax,%r14,8), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x21,0xcd,0x40,0x56,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vorpd  291(%rax,%r14,8), %zmm22, %zmm30

// CHECK: vorpd  (%rcx){1to8}, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x56,0x31]
          vorpd  (%rcx){1to8}, %zmm22, %zmm30

// CHECK: vorpd  8128(%rdx), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x56,0x72,0x7f]
          vorpd  8128(%rdx), %zmm22, %zmm30

// CHECK: vorpd  8192(%rdx), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x56,0xb2,0x00,0x20,0x00,0x00]
          vorpd  8192(%rdx), %zmm22, %zmm30

// CHECK: vorpd  -8192(%rdx), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x56,0x72,0x80]
          vorpd  -8192(%rdx), %zmm22, %zmm30

// CHECK: vorpd  -8256(%rdx), %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x56,0xb2,0xc0,0xdf,0xff,0xff]
          vorpd  -8256(%rdx), %zmm22, %zmm30

// CHECK: vorpd  1016(%rdx){1to8}, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x56,0x72,0x7f]
          vorpd  1016(%rdx){1to8}, %zmm22, %zmm30

// CHECK: vorpd  1024(%rdx){1to8}, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x56,0xb2,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to8}, %zmm22, %zmm30

// CHECK: vorpd  -1024(%rdx){1to8}, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x56,0x72,0x80]
          vorpd  -1024(%rdx){1to8}, %zmm22, %zmm30

// CHECK: vorpd  -1032(%rdx){1to8}, %zmm22, %zmm30
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x56,0xb2,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to8}, %zmm22, %zmm30

// CHECK: vorps  %zmm26, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0x81,0x54,0x40,0x56,0xf2]
          vorps  %zmm26, %zmm21, %zmm22

// CHECK: vorps  %zmm26, %zmm21, %zmm22 {%k7}
// CHECK:  encoding: [0x62,0x81,0x54,0x47,0x56,0xf2]
          vorps  %zmm26, %zmm21, %zmm22 {%k7}

// CHECK: vorps  %zmm26, %zmm21, %zmm22 {%k7} {z}
// CHECK:  encoding: [0x62,0x81,0x54,0xc7,0x56,0xf2]
          vorps  %zmm26, %zmm21, %zmm22 {%k7} {z}

// CHECK: vorps  (%rcx), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x56,0x31]
          vorps  (%rcx), %zmm21, %zmm22

// CHECK: vorps  291(%rax,%r14,8), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xa1,0x54,0x40,0x56,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vorps  291(%rax,%r14,8), %zmm21, %zmm22

// CHECK: vorps  (%rcx){1to16}, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x56,0x31]
          vorps  (%rcx){1to16}, %zmm21, %zmm22

// CHECK: vorps  8128(%rdx), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x56,0x72,0x7f]
          vorps  8128(%rdx), %zmm21, %zmm22

// CHECK: vorps  8192(%rdx), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x56,0xb2,0x00,0x20,0x00,0x00]
          vorps  8192(%rdx), %zmm21, %zmm22

// CHECK: vorps  -8192(%rdx), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x56,0x72,0x80]
          vorps  -8192(%rdx), %zmm21, %zmm22

// CHECK: vorps  -8256(%rdx), %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x56,0xb2,0xc0,0xdf,0xff,0xff]
          vorps  -8256(%rdx), %zmm21, %zmm22

// CHECK: vorps  508(%rdx){1to16}, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x56,0x72,0x7f]
          vorps  508(%rdx){1to16}, %zmm21, %zmm22

// CHECK: vorps  512(%rdx){1to16}, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x56,0xb2,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to16}, %zmm21, %zmm22

// CHECK: vorps  -512(%rdx){1to16}, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x56,0x72,0x80]
          vorps  -512(%rdx){1to16}, %zmm21, %zmm22

// CHECK: vorps  -516(%rdx){1to16}, %zmm21, %zmm22
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x56,0xb2,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to16}, %zmm21, %zmm22

// CHECK: vxorpd %zmm24, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x01,0xbd,0x40,0x57,0xd8]
          vxorpd %zmm24, %zmm24, %zmm27

// CHECK: vxorpd %zmm24, %zmm24, %zmm27 {%k5}
// CHECK:  encoding: [0x62,0x01,0xbd,0x45,0x57,0xd8]
          vxorpd %zmm24, %zmm24, %zmm27 {%k5}

// CHECK: vxorpd %zmm24, %zmm24, %zmm27 {%k5} {z}
// CHECK:  encoding: [0x62,0x01,0xbd,0xc5,0x57,0xd8]
          vxorpd %zmm24, %zmm24, %zmm27 {%k5} {z}

// CHECK: vxorpd (%rcx), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x40,0x57,0x19]
          vxorpd (%rcx), %zmm24, %zmm27

// CHECK: vxorpd 291(%rax,%r14,8), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x21,0xbd,0x40,0x57,0x9c,0xf0,0x23,0x01,0x00,0x00]
          vxorpd 291(%rax,%r14,8), %zmm24, %zmm27

// CHECK: vxorpd (%rcx){1to8}, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x50,0x57,0x19]
          vxorpd (%rcx){1to8}, %zmm24, %zmm27

// CHECK: vxorpd 8128(%rdx), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x40,0x57,0x5a,0x7f]
          vxorpd 8128(%rdx), %zmm24, %zmm27

// CHECK: vxorpd 8192(%rdx), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x40,0x57,0x9a,0x00,0x20,0x00,0x00]
          vxorpd 8192(%rdx), %zmm24, %zmm27

// CHECK: vxorpd -8192(%rdx), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x40,0x57,0x5a,0x80]
          vxorpd -8192(%rdx), %zmm24, %zmm27

// CHECK: vxorpd -8256(%rdx), %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x40,0x57,0x9a,0xc0,0xdf,0xff,0xff]
          vxorpd -8256(%rdx), %zmm24, %zmm27

// CHECK: vxorpd 1016(%rdx){1to8}, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x50,0x57,0x5a,0x7f]
          vxorpd 1016(%rdx){1to8}, %zmm24, %zmm27

// CHECK: vxorpd 1024(%rdx){1to8}, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x50,0x57,0x9a,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to8}, %zmm24, %zmm27

// CHECK: vxorpd -1024(%rdx){1to8}, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x50,0x57,0x5a,0x80]
          vxorpd -1024(%rdx){1to8}, %zmm24, %zmm27

// CHECK: vxorpd -1032(%rdx){1to8}, %zmm24, %zmm27
// CHECK:  encoding: [0x62,0x61,0xbd,0x50,0x57,0x9a,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to8}, %zmm24, %zmm27

// CHECK: vxorps %zmm19, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xa1,0x6c,0x40,0x57,0xd3]
          vxorps %zmm19, %zmm18, %zmm18

// CHECK: vxorps %zmm19, %zmm18, %zmm18 {%k2}
// CHECK:  encoding: [0x62,0xa1,0x6c,0x42,0x57,0xd3]
          vxorps %zmm19, %zmm18, %zmm18 {%k2}

// CHECK: vxorps %zmm19, %zmm18, %zmm18 {%k2} {z}
// CHECK:  encoding: [0x62,0xa1,0x6c,0xc2,0x57,0xd3]
          vxorps %zmm19, %zmm18, %zmm18 {%k2} {z}

// CHECK: vxorps (%rcx), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x40,0x57,0x11]
          vxorps (%rcx), %zmm18, %zmm18

// CHECK: vxorps 291(%rax,%r14,8), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xa1,0x6c,0x40,0x57,0x94,0xf0,0x23,0x01,0x00,0x00]
          vxorps 291(%rax,%r14,8), %zmm18, %zmm18

// CHECK: vxorps (%rcx){1to16}, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x50,0x57,0x11]
          vxorps (%rcx){1to16}, %zmm18, %zmm18

// CHECK: vxorps 8128(%rdx), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x40,0x57,0x52,0x7f]
          vxorps 8128(%rdx), %zmm18, %zmm18

// CHECK: vxorps 8192(%rdx), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x40,0x57,0x92,0x00,0x20,0x00,0x00]
          vxorps 8192(%rdx), %zmm18, %zmm18

// CHECK: vxorps -8192(%rdx), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x40,0x57,0x52,0x80]
          vxorps -8192(%rdx), %zmm18, %zmm18

// CHECK: vxorps -8256(%rdx), %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x40,0x57,0x92,0xc0,0xdf,0xff,0xff]
          vxorps -8256(%rdx), %zmm18, %zmm18

// CHECK: vxorps 508(%rdx){1to16}, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x50,0x57,0x52,0x7f]
          vxorps 508(%rdx){1to16}, %zmm18, %zmm18

// CHECK: vxorps 512(%rdx){1to16}, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x50,0x57,0x92,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to16}, %zmm18, %zmm18

// CHECK: vxorps -512(%rdx){1to16}, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x50,0x57,0x52,0x80]
          vxorps -512(%rdx){1to16}, %zmm18, %zmm18

// CHECK: vxorps -516(%rdx){1to16}, %zmm18, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x6c,0x50,0x57,0x92,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to16}, %zmm18, %zmm18
// CHECK: vandpd %zmm22, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x21,0xcd,0x40,0x54,0xc6]
          vandpd %zmm22, %zmm22, %zmm24

// CHECK: vandpd %zmm22, %zmm22, %zmm24 {%k4}
// CHECK:  encoding: [0x62,0x21,0xcd,0x44,0x54,0xc6]
          vandpd %zmm22, %zmm22, %zmm24 {%k4}

// CHECK: vandpd %zmm22, %zmm22, %zmm24 {%k4} {z}
// CHECK:  encoding: [0x62,0x21,0xcd,0xc4,0x54,0xc6]
          vandpd %zmm22, %zmm22, %zmm24 {%k4} {z}

// CHECK: vandpd (%rcx), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x54,0x01]
          vandpd (%rcx), %zmm22, %zmm24

// CHECK: vandpd 4660(%rax,%r14,8), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x21,0xcd,0x40,0x54,0x84,0xf0,0x34,0x12,0x00,0x00]
          vandpd 4660(%rax,%r14,8), %zmm22, %zmm24

// CHECK: vandpd (%rcx){1to8}, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x54,0x01]
          vandpd (%rcx){1to8}, %zmm22, %zmm24

// CHECK: vandpd 8128(%rdx), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x54,0x42,0x7f]
          vandpd 8128(%rdx), %zmm22, %zmm24

// CHECK: vandpd 8192(%rdx), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x54,0x82,0x00,0x20,0x00,0x00]
          vandpd 8192(%rdx), %zmm22, %zmm24

// CHECK: vandpd -8192(%rdx), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x54,0x42,0x80]
          vandpd -8192(%rdx), %zmm22, %zmm24

// CHECK: vandpd -8256(%rdx), %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x40,0x54,0x82,0xc0,0xdf,0xff,0xff]
          vandpd -8256(%rdx), %zmm22, %zmm24

// CHECK: vandpd 1016(%rdx){1to8}, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x54,0x42,0x7f]
          vandpd 1016(%rdx){1to8}, %zmm22, %zmm24

// CHECK: vandpd 1024(%rdx){1to8}, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x54,0x82,0x00,0x04,0x00,0x00]
          vandpd 1024(%rdx){1to8}, %zmm22, %zmm24

// CHECK: vandpd -1024(%rdx){1to8}, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x54,0x42,0x80]
          vandpd -1024(%rdx){1to8}, %zmm22, %zmm24

// CHECK: vandpd -1032(%rdx){1to8}, %zmm22, %zmm24
// CHECK:  encoding: [0x62,0x61,0xcd,0x50,0x54,0x82,0xf8,0xfb,0xff,0xff]
          vandpd -1032(%rdx){1to8}, %zmm22, %zmm24

// CHECK: vandps %zmm23, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x21,0x44,0x40,0x54,0xf7]
          vandps %zmm23, %zmm23, %zmm30

// CHECK: vandps %zmm23, %zmm23, %zmm30 {%k5}
// CHECK:  encoding: [0x62,0x21,0x44,0x45,0x54,0xf7]
          vandps %zmm23, %zmm23, %zmm30 {%k5}

// CHECK: vandps %zmm23, %zmm23, %zmm30 {%k5} {z}
// CHECK:  encoding: [0x62,0x21,0x44,0xc5,0x54,0xf7]
          vandps %zmm23, %zmm23, %zmm30 {%k5} {z}

// CHECK: vandps (%rcx), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x40,0x54,0x31]
          vandps (%rcx), %zmm23, %zmm30

// CHECK: vandps 4660(%rax,%r14,8), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x21,0x44,0x40,0x54,0xb4,0xf0,0x34,0x12,0x00,0x00]
          vandps 4660(%rax,%r14,8), %zmm23, %zmm30

// CHECK: vandps (%rcx){1to16}, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x50,0x54,0x31]
          vandps (%rcx){1to16}, %zmm23, %zmm30

// CHECK: vandps 8128(%rdx), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x40,0x54,0x72,0x7f]
          vandps 8128(%rdx), %zmm23, %zmm30

// CHECK: vandps 8192(%rdx), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x40,0x54,0xb2,0x00,0x20,0x00,0x00]
          vandps 8192(%rdx), %zmm23, %zmm30

// CHECK: vandps -8192(%rdx), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x40,0x54,0x72,0x80]
          vandps -8192(%rdx), %zmm23, %zmm30

// CHECK: vandps -8256(%rdx), %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x40,0x54,0xb2,0xc0,0xdf,0xff,0xff]
          vandps -8256(%rdx), %zmm23, %zmm30

// CHECK: vandps 508(%rdx){1to16}, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x50,0x54,0x72,0x7f]
          vandps 508(%rdx){1to16}, %zmm23, %zmm30

// CHECK: vandps 512(%rdx){1to16}, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x50,0x54,0xb2,0x00,0x02,0x00,0x00]
          vandps 512(%rdx){1to16}, %zmm23, %zmm30

// CHECK: vandps -512(%rdx){1to16}, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x50,0x54,0x72,0x80]
          vandps -512(%rdx){1to16}, %zmm23, %zmm30

// CHECK: vandps -516(%rdx){1to16}, %zmm23, %zmm30
// CHECK:  encoding: [0x62,0x61,0x44,0x50,0x54,0xb2,0xfc,0xfd,0xff,0xff]
          vandps -516(%rdx){1to16}, %zmm23, %zmm30

// CHECK: vandnpd %zmm21, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x21,0xd5,0x40,0x55,0xcd]
          vandnpd %zmm21, %zmm21, %zmm25

// CHECK: vandnpd %zmm21, %zmm21, %zmm25 {%k2}
// CHECK:  encoding: [0x62,0x21,0xd5,0x42,0x55,0xcd]
          vandnpd %zmm21, %zmm21, %zmm25 {%k2}

// CHECK: vandnpd %zmm21, %zmm21, %zmm25 {%k2} {z}
// CHECK:  encoding: [0x62,0x21,0xd5,0xc2,0x55,0xcd]
          vandnpd %zmm21, %zmm21, %zmm25 {%k2} {z}

// CHECK: vandnpd (%rcx), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x40,0x55,0x09]
          vandnpd (%rcx), %zmm21, %zmm25

// CHECK: vandnpd 4660(%rax,%r14,8), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x21,0xd5,0x40,0x55,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vandnpd 4660(%rax,%r14,8), %zmm21, %zmm25

// CHECK: vandnpd (%rcx){1to8}, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x50,0x55,0x09]
          vandnpd (%rcx){1to8}, %zmm21, %zmm25

// CHECK: vandnpd 8128(%rdx), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x40,0x55,0x4a,0x7f]
          vandnpd 8128(%rdx), %zmm21, %zmm25

// CHECK: vandnpd 8192(%rdx), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x40,0x55,0x8a,0x00,0x20,0x00,0x00]
          vandnpd 8192(%rdx), %zmm21, %zmm25

// CHECK: vandnpd -8192(%rdx), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x40,0x55,0x4a,0x80]
          vandnpd -8192(%rdx), %zmm21, %zmm25

// CHECK: vandnpd -8256(%rdx), %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x40,0x55,0x8a,0xc0,0xdf,0xff,0xff]
          vandnpd -8256(%rdx), %zmm21, %zmm25

// CHECK: vandnpd 1016(%rdx){1to8}, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x50,0x55,0x4a,0x7f]
          vandnpd 1016(%rdx){1to8}, %zmm21, %zmm25

// CHECK: vandnpd 1024(%rdx){1to8}, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x50,0x55,0x8a,0x00,0x04,0x00,0x00]
          vandnpd 1024(%rdx){1to8}, %zmm21, %zmm25

// CHECK: vandnpd -1024(%rdx){1to8}, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x50,0x55,0x4a,0x80]
          vandnpd -1024(%rdx){1to8}, %zmm21, %zmm25

// CHECK: vandnpd -1032(%rdx){1to8}, %zmm21, %zmm25
// CHECK:  encoding: [0x62,0x61,0xd5,0x50,0x55,0x8a,0xf8,0xfb,0xff,0xff]
          vandnpd -1032(%rdx){1to8}, %zmm21, %zmm25

// CHECK: vandnps %zmm18, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xa1,0x54,0x40,0x55,0xca]
          vandnps %zmm18, %zmm21, %zmm17

// CHECK: vandnps %zmm18, %zmm21, %zmm17 {%k1}
// CHECK:  encoding: [0x62,0xa1,0x54,0x41,0x55,0xca]
          vandnps %zmm18, %zmm21, %zmm17 {%k1}

// CHECK: vandnps %zmm18, %zmm21, %zmm17 {%k1} {z}
// CHECK:  encoding: [0x62,0xa1,0x54,0xc1,0x55,0xca]
          vandnps %zmm18, %zmm21, %zmm17 {%k1} {z}

// CHECK: vandnps (%rcx), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x55,0x09]
          vandnps (%rcx), %zmm21, %zmm17

// CHECK: vandnps 4660(%rax,%r14,8), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xa1,0x54,0x40,0x55,0x8c,0xf0,0x34,0x12,0x00,0x00]
          vandnps 4660(%rax,%r14,8), %zmm21, %zmm17

// CHECK: vandnps (%rcx){1to16}, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x55,0x09]
          vandnps (%rcx){1to16}, %zmm21, %zmm17

// CHECK: vandnps 8128(%rdx), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x55,0x4a,0x7f]
          vandnps 8128(%rdx), %zmm21, %zmm17

// CHECK: vandnps 8192(%rdx), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x55,0x8a,0x00,0x20,0x00,0x00]
          vandnps 8192(%rdx), %zmm21, %zmm17

// CHECK: vandnps -8192(%rdx), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x55,0x4a,0x80]
          vandnps -8192(%rdx), %zmm21, %zmm17

// CHECK: vandnps -8256(%rdx), %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x40,0x55,0x8a,0xc0,0xdf,0xff,0xff]
          vandnps -8256(%rdx), %zmm21, %zmm17

// CHECK: vandnps 508(%rdx){1to16}, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x55,0x4a,0x7f]
          vandnps 508(%rdx){1to16}, %zmm21, %zmm17

// CHECK: vandnps 512(%rdx){1to16}, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x55,0x8a,0x00,0x02,0x00,0x00]
          vandnps 512(%rdx){1to16}, %zmm21, %zmm17

// CHECK: vandnps -512(%rdx){1to16}, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x55,0x4a,0x80]
          vandnps -512(%rdx){1to16}, %zmm21, %zmm17

// CHECK: vandnps -516(%rdx){1to16}, %zmm21, %zmm17
// CHECK:  encoding: [0x62,0xe1,0x54,0x50,0x55,0x8a,0xfc,0xfd,0xff,0xff]
          vandnps -516(%rdx){1to16}, %zmm21, %zmm17

// CHECK: vorpd  %zmm24, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0x81,0x9d,0x40,0x56,0xd0]
          vorpd  %zmm24, %zmm28, %zmm18

// CHECK: vorpd  %zmm24, %zmm28, %zmm18 {%k1}
// CHECK:  encoding: [0x62,0x81,0x9d,0x41,0x56,0xd0]
          vorpd  %zmm24, %zmm28, %zmm18 {%k1}

// CHECK: vorpd  %zmm24, %zmm28, %zmm18 {%k1} {z}
// CHECK:  encoding: [0x62,0x81,0x9d,0xc1,0x56,0xd0]
          vorpd  %zmm24, %zmm28, %zmm18 {%k1} {z}

// CHECK: vorpd  (%rcx), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x56,0x11]
          vorpd  (%rcx), %zmm28, %zmm18

// CHECK: vorpd  4660(%rax,%r14,8), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xa1,0x9d,0x40,0x56,0x94,0xf0,0x34,0x12,0x00,0x00]
          vorpd  4660(%rax,%r14,8), %zmm28, %zmm18

// CHECK: vorpd  (%rcx){1to8}, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x56,0x11]
          vorpd  (%rcx){1to8}, %zmm28, %zmm18

// CHECK: vorpd  8128(%rdx), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x56,0x52,0x7f]
          vorpd  8128(%rdx), %zmm28, %zmm18

// CHECK: vorpd  8192(%rdx), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x56,0x92,0x00,0x20,0x00,0x00]
          vorpd  8192(%rdx), %zmm28, %zmm18

// CHECK: vorpd  -8192(%rdx), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x56,0x52,0x80]
          vorpd  -8192(%rdx), %zmm28, %zmm18

// CHECK: vorpd  -8256(%rdx), %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x40,0x56,0x92,0xc0,0xdf,0xff,0xff]
          vorpd  -8256(%rdx), %zmm28, %zmm18

// CHECK: vorpd  1016(%rdx){1to8}, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x56,0x52,0x7f]
          vorpd  1016(%rdx){1to8}, %zmm28, %zmm18

// CHECK: vorpd  1024(%rdx){1to8}, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x56,0x92,0x00,0x04,0x00,0x00]
          vorpd  1024(%rdx){1to8}, %zmm28, %zmm18

// CHECK: vorpd  -1024(%rdx){1to8}, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x56,0x52,0x80]
          vorpd  -1024(%rdx){1to8}, %zmm28, %zmm18

// CHECK: vorpd  -1032(%rdx){1to8}, %zmm28, %zmm18
// CHECK:  encoding: [0x62,0xe1,0x9d,0x50,0x56,0x92,0xf8,0xfb,0xff,0xff]
          vorpd  -1032(%rdx){1to8}, %zmm28, %zmm18

// CHECK: vorps  %zmm23, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x21,0x74,0x40,0x56,0xe7]
          vorps  %zmm23, %zmm17, %zmm28

// CHECK: vorps  %zmm23, %zmm17, %zmm28 {%k7}
// CHECK:  encoding: [0x62,0x21,0x74,0x47,0x56,0xe7]
          vorps  %zmm23, %zmm17, %zmm28 {%k7}

// CHECK: vorps  %zmm23, %zmm17, %zmm28 {%k7} {z}
// CHECK:  encoding: [0x62,0x21,0x74,0xc7,0x56,0xe7]
          vorps  %zmm23, %zmm17, %zmm28 {%k7} {z}

// CHECK: vorps  (%rcx), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x40,0x56,0x21]
          vorps  (%rcx), %zmm17, %zmm28

// CHECK: vorps  4660(%rax,%r14,8), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x21,0x74,0x40,0x56,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vorps  4660(%rax,%r14,8), %zmm17, %zmm28

// CHECK: vorps  (%rcx){1to16}, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x50,0x56,0x21]
          vorps  (%rcx){1to16}, %zmm17, %zmm28

// CHECK: vorps  8128(%rdx), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x40,0x56,0x62,0x7f]
          vorps  8128(%rdx), %zmm17, %zmm28

// CHECK: vorps  8192(%rdx), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x40,0x56,0xa2,0x00,0x20,0x00,0x00]
          vorps  8192(%rdx), %zmm17, %zmm28

// CHECK: vorps  -8192(%rdx), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x40,0x56,0x62,0x80]
          vorps  -8192(%rdx), %zmm17, %zmm28

// CHECK: vorps  -8256(%rdx), %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x40,0x56,0xa2,0xc0,0xdf,0xff,0xff]
          vorps  -8256(%rdx), %zmm17, %zmm28

// CHECK: vorps  508(%rdx){1to16}, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x50,0x56,0x62,0x7f]
          vorps  508(%rdx){1to16}, %zmm17, %zmm28

// CHECK: vorps  512(%rdx){1to16}, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x50,0x56,0xa2,0x00,0x02,0x00,0x00]
          vorps  512(%rdx){1to16}, %zmm17, %zmm28

// CHECK: vorps  -512(%rdx){1to16}, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x50,0x56,0x62,0x80]
          vorps  -512(%rdx){1to16}, %zmm17, %zmm28

// CHECK: vorps  -516(%rdx){1to16}, %zmm17, %zmm28
// CHECK:  encoding: [0x62,0x61,0x74,0x50,0x56,0xa2,0xfc,0xfd,0xff,0xff]
          vorps  -516(%rdx){1to16}, %zmm17, %zmm28

// CHECK: vxorpd %zmm27, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x01,0xed,0x40,0x57,0xe3]
          vxorpd %zmm27, %zmm18, %zmm28

// CHECK: vxorpd %zmm27, %zmm18, %zmm28 {%k4}
// CHECK:  encoding: [0x62,0x01,0xed,0x44,0x57,0xe3]
          vxorpd %zmm27, %zmm18, %zmm28 {%k4}

// CHECK: vxorpd %zmm27, %zmm18, %zmm28 {%k4} {z}
// CHECK:  encoding: [0x62,0x01,0xed,0xc4,0x57,0xe3]
          vxorpd %zmm27, %zmm18, %zmm28 {%k4} {z}

// CHECK: vxorpd (%rcx), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x40,0x57,0x21]
          vxorpd (%rcx), %zmm18, %zmm28

// CHECK: vxorpd 4660(%rax,%r14,8), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x21,0xed,0x40,0x57,0xa4,0xf0,0x34,0x12,0x00,0x00]
          vxorpd 4660(%rax,%r14,8), %zmm18, %zmm28

// CHECK: vxorpd (%rcx){1to8}, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x50,0x57,0x21]
          vxorpd (%rcx){1to8}, %zmm18, %zmm28

// CHECK: vxorpd 8128(%rdx), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x40,0x57,0x62,0x7f]
          vxorpd 8128(%rdx), %zmm18, %zmm28

// CHECK: vxorpd 8192(%rdx), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x40,0x57,0xa2,0x00,0x20,0x00,0x00]
          vxorpd 8192(%rdx), %zmm18, %zmm28

// CHECK: vxorpd -8192(%rdx), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x40,0x57,0x62,0x80]
          vxorpd -8192(%rdx), %zmm18, %zmm28

// CHECK: vxorpd -8256(%rdx), %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x40,0x57,0xa2,0xc0,0xdf,0xff,0xff]
          vxorpd -8256(%rdx), %zmm18, %zmm28

// CHECK: vxorpd 1016(%rdx){1to8}, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x50,0x57,0x62,0x7f]
          vxorpd 1016(%rdx){1to8}, %zmm18, %zmm28

// CHECK: vxorpd 1024(%rdx){1to8}, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x50,0x57,0xa2,0x00,0x04,0x00,0x00]
          vxorpd 1024(%rdx){1to8}, %zmm18, %zmm28

// CHECK: vxorpd -1024(%rdx){1to8}, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x50,0x57,0x62,0x80]
          vxorpd -1024(%rdx){1to8}, %zmm18, %zmm28

// CHECK: vxorpd -1032(%rdx){1to8}, %zmm18, %zmm28
// CHECK:  encoding: [0x62,0x61,0xed,0x50,0x57,0xa2,0xf8,0xfb,0xff,0xff]
          vxorpd -1032(%rdx){1to8}, %zmm18, %zmm28

// CHECK: vxorps %zmm18, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x21,0x1c,0x40,0x57,0xc2]
          vxorps %zmm18, %zmm28, %zmm24

// CHECK: vxorps %zmm18, %zmm28, %zmm24 {%k4}
// CHECK:  encoding: [0x62,0x21,0x1c,0x44,0x57,0xc2]
          vxorps %zmm18, %zmm28, %zmm24 {%k4}

// CHECK: vxorps %zmm18, %zmm28, %zmm24 {%k4} {z}
// CHECK:  encoding: [0x62,0x21,0x1c,0xc4,0x57,0xc2]
          vxorps %zmm18, %zmm28, %zmm24 {%k4} {z}

// CHECK: vxorps (%rcx), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x40,0x57,0x01]
          vxorps (%rcx), %zmm28, %zmm24

// CHECK: vxorps 4660(%rax,%r14,8), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x21,0x1c,0x40,0x57,0x84,0xf0,0x34,0x12,0x00,0x00]
          vxorps 4660(%rax,%r14,8), %zmm28, %zmm24

// CHECK: vxorps (%rcx){1to16}, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x50,0x57,0x01]
          vxorps (%rcx){1to16}, %zmm28, %zmm24

// CHECK: vxorps 8128(%rdx), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x40,0x57,0x42,0x7f]
          vxorps 8128(%rdx), %zmm28, %zmm24

// CHECK: vxorps 8192(%rdx), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x40,0x57,0x82,0x00,0x20,0x00,0x00]
          vxorps 8192(%rdx), %zmm28, %zmm24

// CHECK: vxorps -8192(%rdx), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x40,0x57,0x42,0x80]
          vxorps -8192(%rdx), %zmm28, %zmm24

// CHECK: vxorps -8256(%rdx), %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x40,0x57,0x82,0xc0,0xdf,0xff,0xff]
          vxorps -8256(%rdx), %zmm28, %zmm24

// CHECK: vxorps 508(%rdx){1to16}, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x50,0x57,0x42,0x7f]
          vxorps 508(%rdx){1to16}, %zmm28, %zmm24

// CHECK: vxorps 512(%rdx){1to16}, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x50,0x57,0x82,0x00,0x02,0x00,0x00]
          vxorps 512(%rdx){1to16}, %zmm28, %zmm24

// CHECK: vxorps -512(%rdx){1to16}, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x50,0x57,0x42,0x80]
          vxorps -512(%rdx){1to16}, %zmm28, %zmm24

// CHECK: vxorps -516(%rdx){1to16}, %zmm28, %zmm24
// CHECK:  encoding: [0x62,0x61,0x1c,0x50,0x57,0x82,0xfc,0xfd,0xff,0xff]
          vxorps -516(%rdx){1to16}, %zmm28, %zmm24

// CHECK: vinserti32x8
// CHECK: encoding: [0x62,0xd3,0x4d,0x40,0x3a,0xdb,0x01]
          vinserti32x8  $1, %ymm11, %zmm22, %zmm3

// CHECK: vinsertf64x2
// CHECK: encoding: [0x62,0xf3,0xed,0x48,0x18,0x4f,0x10,0x01]
          vinsertf64x2  $1, 256(%rdi), %zmm2, %zmm1

// CHECK: vbroadcastf32x8 (%rcx), %zmm30
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x1b,0x31]
          vbroadcastf32x8 (%rcx), %zmm30

// CHECK: vbroadcastf32x8 (%rcx), %zmm30 {%k3}
// CHECK:  encoding: [0x62,0x62,0x7d,0x4b,0x1b,0x31]
          vbroadcastf32x8 (%rcx), %zmm30 {%k3}

// CHECK: vbroadcastf32x8 (%rcx), %zmm30 {%k3} {z}
// CHECK:  encoding: [0x62,0x62,0x7d,0xcb,0x1b,0x31]
          vbroadcastf32x8 (%rcx), %zmm30 {%k3} {z}

// CHECK: vbroadcastf32x8 291(%rax,%r14,8), %zmm30
// CHECK:  encoding: [0x62,0x22,0x7d,0x48,0x1b,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vbroadcastf32x8 291(%rax,%r14,8), %zmm30

// CHECK: vbroadcastf32x8 4064(%rdx), %zmm30
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x1b,0x72,0x7f]
          vbroadcastf32x8 4064(%rdx), %zmm30

// CHECK: vbroadcastf32x8 4096(%rdx), %zmm30
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x1b,0xb2,0x00,0x10,0x00,0x00]
          vbroadcastf32x8 4096(%rdx), %zmm30

// CHECK: vbroadcastf32x8 -4096(%rdx), %zmm30
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x1b,0x72,0x80]
          vbroadcastf32x8 -4096(%rdx), %zmm30

// CHECK: vbroadcastf32x8 -4128(%rdx), %zmm30
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x1b,0xb2,0xe0,0xef,0xff,0xff]
          vbroadcastf32x8 -4128(%rdx), %zmm30

// CHECK: vbroadcastf64x2 (%rcx), %zmm28
// CHECK:  encoding: [0x62,0x62,0xfd,0x48,0x1a,0x21]
          vbroadcastf64x2 (%rcx), %zmm28

// CHECK: vbroadcastf64x2 (%rcx), %zmm28 {%k4}
// CHECK:  encoding: [0x62,0x62,0xfd,0x4c,0x1a,0x21]
          vbroadcastf64x2 (%rcx), %zmm28 {%k4}

// CHECK: vbroadcastf64x2 (%rcx), %zmm28 {%k4} {z}
// CHECK:  encoding: [0x62,0x62,0xfd,0xcc,0x1a,0x21]
          vbroadcastf64x2 (%rcx), %zmm28 {%k4} {z}

// CHECK: vbroadcastf64x2 291(%rax,%r14,8), %zmm28
// CHECK:  encoding: [0x62,0x22,0xfd,0x48,0x1a,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vbroadcastf64x2 291(%rax,%r14,8), %zmm28

// CHECK: vbroadcastf64x2 2032(%rdx), %zmm28
// CHECK:  encoding: [0x62,0x62,0xfd,0x48,0x1a,0x62,0x7f]
          vbroadcastf64x2 2032(%rdx), %zmm28

// CHECK: vbroadcastf64x2 2048(%rdx), %zmm28
// CHECK:  encoding: [0x62,0x62,0xfd,0x48,0x1a,0xa2,0x00,0x08,0x00,0x00]
          vbroadcastf64x2 2048(%rdx), %zmm28

// CHECK: vbroadcastf64x2 -2048(%rdx), %zmm28
// CHECK:  encoding: [0x62,0x62,0xfd,0x48,0x1a,0x62,0x80]
          vbroadcastf64x2 -2048(%rdx), %zmm28

// CHECK: vbroadcastf64x2 -2064(%rdx), %zmm28
// CHECK:  encoding: [0x62,0x62,0xfd,0x48,0x1a,0xa2,0xf0,0xf7,0xff,0xff]
          vbroadcastf64x2 -2064(%rdx), %zmm28

// CHECK: vbroadcasti32x8 (%rcx), %zmm29
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x5b,0x29]
          vbroadcasti32x8 (%rcx), %zmm29

// CHECK: vbroadcasti32x8 (%rcx), %zmm29 {%k5}
// CHECK:  encoding: [0x62,0x62,0x7d,0x4d,0x5b,0x29]
          vbroadcasti32x8 (%rcx), %zmm29 {%k5}

// CHECK: vbroadcasti32x8 (%rcx), %zmm29 {%k5} {z}
// CHECK:  encoding: [0x62,0x62,0x7d,0xcd,0x5b,0x29]
          vbroadcasti32x8 (%rcx), %zmm29 {%k5} {z}

// CHECK: vbroadcasti32x8 291(%rax,%r14,8), %zmm29
// CHECK:  encoding: [0x62,0x22,0x7d,0x48,0x5b,0xac,0xf0,0x23,0x01,0x00,0x00]
          vbroadcasti32x8 291(%rax,%r14,8), %zmm29

// CHECK: vbroadcasti32x8 4064(%rdx), %zmm29
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x5b,0x6a,0x7f]
          vbroadcasti32x8 4064(%rdx), %zmm29

// CHECK: vbroadcasti32x8 4096(%rdx), %zmm29
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x5b,0xaa,0x00,0x10,0x00,0x00]
          vbroadcasti32x8 4096(%rdx), %zmm29

// CHECK: vbroadcasti32x8 -4096(%rdx), %zmm29
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x5b,0x6a,0x80]
          vbroadcasti32x8 -4096(%rdx), %zmm29

// CHECK: vbroadcasti32x8 -4128(%rdx), %zmm29
// CHECK:  encoding: [0x62,0x62,0x7d,0x48,0x5b,0xaa,0xe0,0xef,0xff,0xff]
          vbroadcasti32x8 -4128(%rdx), %zmm29

// CHECK: vbroadcasti64x2 (%rcx), %zmm20
// CHECK:  encoding: [0x62,0xe2,0xfd,0x48,0x5a,0x21]
          vbroadcasti64x2 (%rcx), %zmm20

// CHECK: vbroadcasti64x2 (%rcx), %zmm20 {%k3}
// CHECK:  encoding: [0x62,0xe2,0xfd,0x4b,0x5a,0x21]
          vbroadcasti64x2 (%rcx), %zmm20 {%k3}

// CHECK: vbroadcasti64x2 (%rcx), %zmm20 {%k3} {z}
// CHECK:  encoding: [0x62,0xe2,0xfd,0xcb,0x5a,0x21]
          vbroadcasti64x2 (%rcx), %zmm20 {%k3} {z}

// CHECK: vbroadcasti64x2 291(%rax,%r14,8), %zmm20
// CHECK:  encoding: [0x62,0xa2,0xfd,0x48,0x5a,0xa4,0xf0,0x23,0x01,0x00,0x00]
          vbroadcasti64x2 291(%rax,%r14,8), %zmm20

// CHECK: vbroadcasti64x2 2032(%rdx), %zmm20
// CHECK:  encoding: [0x62,0xe2,0xfd,0x48,0x5a,0x62,0x7f]
          vbroadcasti64x2 2032(%rdx), %zmm20

// CHECK: vbroadcasti64x2 2048(%rdx), %zmm20
// CHECK:  encoding: [0x62,0xe2,0xfd,0x48,0x5a,0xa2,0x00,0x08,0x00,0x00]
          vbroadcasti64x2 2048(%rdx), %zmm20

// CHECK: vbroadcasti64x2 -2048(%rdx), %zmm20
// CHECK:  encoding: [0x62,0xe2,0xfd,0x48,0x5a,0x62,0x80]
          vbroadcasti64x2 -2048(%rdx), %zmm20

// CHECK: vbroadcasti64x2 -2064(%rdx), %zmm20
// CHECK:  encoding: [0x62,0xe2,0xfd,0x48,0x5a,0xa2,0xf0,0xf7,0xff,0xff]
          vbroadcasti64x2 -2064(%rdx), %zmm20
