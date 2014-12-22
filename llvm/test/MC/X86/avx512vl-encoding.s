// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx  --show-encoding %s | FileCheck %s

// CHECK: vblendmpd %xmm19, %xmm20, %xmm27
          vblendmpd %xmm19, %xmm20, %xmm27
// CHECK: vblendmpd %xmm19, %xmm20, %xmm27 {%k7}
          vblendmpd %xmm19, %xmm20, %xmm27 {%k7}
// CHECK: vblendmpd %xmm19, %xmm20, %xmm27 {%k7} {z}
          vblendmpd %xmm19, %xmm20, %xmm27 {%k7} {z}
// CHECK: vblendmpd (%rcx), %xmm20, %xmm27
          vblendmpd (%rcx), %xmm20, %xmm27
// CHECK: vblendmpd 291(%rax,%r14,8), %xmm20, %xmm27
          vblendmpd 291(%rax,%r14,8), %xmm20, %xmm27
// CHECK: vblendmpd (%rcx){1to2}, %xmm20, %xmm27
          vblendmpd (%rcx){1to2}, %xmm20, %xmm27
// CHECK: vblendmpd 2032(%rdx), %xmm20, %xmm27
          vblendmpd 2032(%rdx), %xmm20, %xmm27
// CHECK: vblendmpd 2048(%rdx), %xmm20, %xmm27
          vblendmpd 2048(%rdx), %xmm20, %xmm27
// CHECK: vblendmpd -2048(%rdx), %xmm20, %xmm27
          vblendmpd -2048(%rdx), %xmm20, %xmm27
// CHECK: vblendmpd -2064(%rdx), %xmm20, %xmm27
          vblendmpd -2064(%rdx), %xmm20, %xmm27
// CHECK: vblendmpd 1016(%rdx){1to2}, %xmm20, %xmm27
          vblendmpd 1016(%rdx){1to2}, %xmm20, %xmm27
// CHECK: vblendmpd 1024(%rdx){1to2}, %xmm20, %xmm27
          vblendmpd 1024(%rdx){1to2}, %xmm20, %xmm27
// CHECK: vblendmpd -1024(%rdx){1to2}, %xmm20, %xmm27
          vblendmpd -1024(%rdx){1to2}, %xmm20, %xmm27
// CHECK: vblendmpd -1032(%rdx){1to2}, %xmm20, %xmm27
          vblendmpd -1032(%rdx){1to2}, %xmm20, %xmm27
// CHECK: vblendmpd %ymm23, %ymm21, %ymm28
          vblendmpd %ymm23, %ymm21, %ymm28
// CHECK: vblendmpd %ymm23, %ymm21, %ymm28 {%k3}
          vblendmpd %ymm23, %ymm21, %ymm28 {%k3}
// CHECK: vblendmpd %ymm23, %ymm21, %ymm28 {%k3} {z}
          vblendmpd %ymm23, %ymm21, %ymm28 {%k3} {z}
// CHECK: vblendmpd (%rcx), %ymm21, %ymm28
          vblendmpd (%rcx), %ymm21, %ymm28
// CHECK: vblendmpd 291(%rax,%r14,8), %ymm21, %ymm28
          vblendmpd 291(%rax,%r14,8), %ymm21, %ymm28
// CHECK: vblendmpd (%rcx){1to4}, %ymm21, %ymm28
          vblendmpd (%rcx){1to4}, %ymm21, %ymm28
// CHECK: vblendmpd 4064(%rdx), %ymm21, %ymm28
          vblendmpd 4064(%rdx), %ymm21, %ymm28
// CHECK: vblendmpd 4096(%rdx), %ymm21, %ymm28
          vblendmpd 4096(%rdx), %ymm21, %ymm28
// CHECK: vblendmpd -4096(%rdx), %ymm21, %ymm28
          vblendmpd -4096(%rdx), %ymm21, %ymm28
// CHECK: vblendmpd -4128(%rdx), %ymm21, %ymm28
          vblendmpd -4128(%rdx), %ymm21, %ymm28
// CHECK: vblendmpd 1016(%rdx){1to4}, %ymm21, %ymm28
          vblendmpd 1016(%rdx){1to4}, %ymm21, %ymm28
// CHECK: vblendmpd 1024(%rdx){1to4}, %ymm21, %ymm28
          vblendmpd 1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK: vblendmpd -1024(%rdx){1to4}, %ymm21, %ymm28
          vblendmpd -1024(%rdx){1to4}, %ymm21, %ymm28
// CHECK: vblendmpd -1032(%rdx){1to4}, %ymm21, %ymm28
          vblendmpd -1032(%rdx){1to4}, %ymm21, %ymm28
// CHECK: vblendmps %xmm20, %xmm20, %xmm24
          vblendmps %xmm20, %xmm20, %xmm24
// CHECK: vblendmps %xmm20, %xmm20, %xmm24 {%k1}
          vblendmps %xmm20, %xmm20, %xmm24 {%k1}
// CHECK: vblendmps %xmm20, %xmm20, %xmm24 {%k1} {z}
          vblendmps %xmm20, %xmm20, %xmm24 {%k1} {z}
// CHECK: vblendmps (%rcx), %xmm20, %xmm24
          vblendmps (%rcx), %xmm20, %xmm24
// CHECK: vblendmps 291(%rax,%r14,8), %xmm20, %xmm24
          vblendmps 291(%rax,%r14,8), %xmm20, %xmm24
// CHECK: vblendmps (%rcx){1to4}, %xmm20, %xmm24
          vblendmps (%rcx){1to4}, %xmm20, %xmm24
// CHECK: vblendmps 2032(%rdx), %xmm20, %xmm24
          vblendmps 2032(%rdx), %xmm20, %xmm24
// CHECK: vblendmps 2048(%rdx), %xmm20, %xmm24
          vblendmps 2048(%rdx), %xmm20, %xmm24
// CHECK: vblendmps -2048(%rdx), %xmm20, %xmm24
          vblendmps -2048(%rdx), %xmm20, %xmm24
// CHECK: vblendmps -2064(%rdx), %xmm20, %xmm24
          vblendmps -2064(%rdx), %xmm20, %xmm24
// CHECK: vblendmps 508(%rdx){1to4}, %xmm20, %xmm24
          vblendmps 508(%rdx){1to4}, %xmm20, %xmm24
// CHECK: vblendmps 512(%rdx){1to4}, %xmm20, %xmm24
          vblendmps 512(%rdx){1to4}, %xmm20, %xmm24
// CHECK: vblendmps -512(%rdx){1to4}, %xmm20, %xmm24
          vblendmps -512(%rdx){1to4}, %xmm20, %xmm24
// CHECK: vblendmps -516(%rdx){1to4}, %xmm20, %xmm24
          vblendmps -516(%rdx){1to4}, %xmm20, %xmm24
// CHECK: vblendmps %ymm24, %ymm23, %ymm17
          vblendmps %ymm24, %ymm23, %ymm17
// CHECK: vblendmps %ymm24, %ymm23, %ymm17 {%k6}
          vblendmps %ymm24, %ymm23, %ymm17 {%k6}
// CHECK: vblendmps %ymm24, %ymm23, %ymm17 {%k6} {z}
          vblendmps %ymm24, %ymm23, %ymm17 {%k6} {z}
// CHECK: vblendmps (%rcx), %ymm23, %ymm17
          vblendmps (%rcx), %ymm23, %ymm17
// CHECK: vblendmps 291(%rax,%r14,8), %ymm23, %ymm17
          vblendmps 291(%rax,%r14,8), %ymm23, %ymm17
// CHECK: vblendmps (%rcx){1to8}, %ymm23, %ymm17
          vblendmps (%rcx){1to8}, %ymm23, %ymm17
// CHECK: vblendmps 4064(%rdx), %ymm23, %ymm17
          vblendmps 4064(%rdx), %ymm23, %ymm17
// CHECK: vblendmps 4096(%rdx), %ymm23, %ymm17
          vblendmps 4096(%rdx), %ymm23, %ymm17
// CHECK: vblendmps -4096(%rdx), %ymm23, %ymm17
          vblendmps -4096(%rdx), %ymm23, %ymm17
// CHECK: vblendmps -4128(%rdx), %ymm23, %ymm17
          vblendmps -4128(%rdx), %ymm23, %ymm17
// CHECK: vblendmps 508(%rdx){1to8}, %ymm23, %ymm17
          vblendmps 508(%rdx){1to8}, %ymm23, %ymm17
// CHECK: vblendmps 512(%rdx){1to8}, %ymm23, %ymm17
          vblendmps 512(%rdx){1to8}, %ymm23, %ymm17
// CHECK: vblendmps -512(%rdx){1to8}, %ymm23, %ymm17
          vblendmps -512(%rdx){1to8}, %ymm23, %ymm17
// CHECK: vblendmps -516(%rdx){1to8}, %ymm23, %ymm17
          vblendmps -516(%rdx){1to8}, %ymm23, %ymm17
// CHECK: vpblendmd %xmm26, %xmm25, %xmm17
          vpblendmd %xmm26, %xmm25, %xmm17
// CHECK: vpblendmd %xmm26, %xmm25, %xmm17 {%k5}
          vpblendmd %xmm26, %xmm25, %xmm17 {%k5}
// CHECK: vpblendmd %xmm26, %xmm25, %xmm17 {%k5} {z}
          vpblendmd %xmm26, %xmm25, %xmm17 {%k5} {z}
// CHECK: vpblendmd (%rcx), %xmm25, %xmm17
          vpblendmd (%rcx), %xmm25, %xmm17
// CHECK: vpblendmd 291(%rax,%r14,8), %xmm25, %xmm17
          vpblendmd 291(%rax,%r14,8), %xmm25, %xmm17
// CHECK: vpblendmd (%rcx){1to4}, %xmm25, %xmm17
          vpblendmd (%rcx){1to4}, %xmm25, %xmm17
// CHECK: vpblendmd 2032(%rdx), %xmm25, %xmm17
          vpblendmd 2032(%rdx), %xmm25, %xmm17
// CHECK: vpblendmd 2048(%rdx), %xmm25, %xmm17
          vpblendmd 2048(%rdx), %xmm25, %xmm17
// CHECK: vpblendmd -2048(%rdx), %xmm25, %xmm17
          vpblendmd -2048(%rdx), %xmm25, %xmm17
// CHECK: vpblendmd -2064(%rdx), %xmm25, %xmm17
          vpblendmd -2064(%rdx), %xmm25, %xmm17
// CHECK: vpblendmd 508(%rdx){1to4}, %xmm25, %xmm17
          vpblendmd 508(%rdx){1to4}, %xmm25, %xmm17
// CHECK: vpblendmd 512(%rdx){1to4}, %xmm25, %xmm17
          vpblendmd 512(%rdx){1to4}, %xmm25, %xmm17
// CHECK: vpblendmd -512(%rdx){1to4}, %xmm25, %xmm17
          vpblendmd -512(%rdx){1to4}, %xmm25, %xmm17
// CHECK: vpblendmd -516(%rdx){1to4}, %xmm25, %xmm17
          vpblendmd -516(%rdx){1to4}, %xmm25, %xmm17
// CHECK: vpblendmd %ymm23, %ymm29, %ymm26
          vpblendmd %ymm23, %ymm29, %ymm26
// CHECK: vpblendmd %ymm23, %ymm29, %ymm26 {%k7}
          vpblendmd %ymm23, %ymm29, %ymm26 {%k7}
// CHECK: vpblendmd %ymm23, %ymm29, %ymm26 {%k7} {z}
          vpblendmd %ymm23, %ymm29, %ymm26 {%k7} {z}
// CHECK: vpblendmd (%rcx), %ymm29, %ymm26
          vpblendmd (%rcx), %ymm29, %ymm26
// CHECK: vpblendmd 291(%rax,%r14,8), %ymm29, %ymm26
          vpblendmd 291(%rax,%r14,8), %ymm29, %ymm26
// CHECK: vpblendmd (%rcx){1to8}, %ymm29, %ymm26
          vpblendmd (%rcx){1to8}, %ymm29, %ymm26
// CHECK: vpblendmd 4064(%rdx), %ymm29, %ymm26
          vpblendmd 4064(%rdx), %ymm29, %ymm26
// CHECK: vpblendmd 4096(%rdx), %ymm29, %ymm26
          vpblendmd 4096(%rdx), %ymm29, %ymm26
// CHECK: vpblendmd -4096(%rdx), %ymm29, %ymm26
          vpblendmd -4096(%rdx), %ymm29, %ymm26
// CHECK: vpblendmd -4128(%rdx), %ymm29, %ymm26
          vpblendmd -4128(%rdx), %ymm29, %ymm26
// CHECK: vpblendmd 508(%rdx){1to8}, %ymm29, %ymm26
          vpblendmd 508(%rdx){1to8}, %ymm29, %ymm26
// CHECK: vpblendmd 512(%rdx){1to8}, %ymm29, %ymm26
          vpblendmd 512(%rdx){1to8}, %ymm29, %ymm26
// CHECK: vpblendmd -512(%rdx){1to8}, %ymm29, %ymm26
          vpblendmd -512(%rdx){1to8}, %ymm29, %ymm26
// CHECK: vpblendmd -516(%rdx){1to8}, %ymm29, %ymm26
          vpblendmd -516(%rdx){1to8}, %ymm29, %ymm26
// CHECK: vpblendmq %xmm17, %xmm27, %xmm29
          vpblendmq %xmm17, %xmm27, %xmm29
// CHECK: vpblendmq %xmm17, %xmm27, %xmm29 {%k6}
          vpblendmq %xmm17, %xmm27, %xmm29 {%k6}
// CHECK: vpblendmq %xmm17, %xmm27, %xmm29 {%k6} {z}
          vpblendmq %xmm17, %xmm27, %xmm29 {%k6} {z}
// CHECK: vpblendmq (%rcx), %xmm27, %xmm29
          vpblendmq (%rcx), %xmm27, %xmm29
// CHECK: vpblendmq 291(%rax,%r14,8), %xmm27, %xmm29
          vpblendmq 291(%rax,%r14,8), %xmm27, %xmm29
// CHECK: vpblendmq (%rcx){1to2}, %xmm27, %xmm29
          vpblendmq (%rcx){1to2}, %xmm27, %xmm29
// CHECK: vpblendmq 2032(%rdx), %xmm27, %xmm29
          vpblendmq 2032(%rdx), %xmm27, %xmm29
// CHECK: vpblendmq 2048(%rdx), %xmm27, %xmm29
          vpblendmq 2048(%rdx), %xmm27, %xmm29
// CHECK: vpblendmq -2048(%rdx), %xmm27, %xmm29
          vpblendmq -2048(%rdx), %xmm27, %xmm29
// CHECK: vpblendmq -2064(%rdx), %xmm27, %xmm29
          vpblendmq -2064(%rdx), %xmm27, %xmm29
// CHECK: vpblendmq 1016(%rdx){1to2}, %xmm27, %xmm29
          vpblendmq 1016(%rdx){1to2}, %xmm27, %xmm29
// CHECK: vpblendmq 1024(%rdx){1to2}, %xmm27, %xmm29
          vpblendmq 1024(%rdx){1to2}, %xmm27, %xmm29
// CHECK: vpblendmq -1024(%rdx){1to2}, %xmm27, %xmm29
          vpblendmq -1024(%rdx){1to2}, %xmm27, %xmm29
// CHECK: vpblendmq -1032(%rdx){1to2}, %xmm27, %xmm29
          vpblendmq -1032(%rdx){1to2}, %xmm27, %xmm29
// CHECK: vpblendmq %ymm21, %ymm23, %ymm21
          vpblendmq %ymm21, %ymm23, %ymm21
// CHECK: vpblendmq %ymm21, %ymm23, %ymm21 {%k3}
          vpblendmq %ymm21, %ymm23, %ymm21 {%k3}
// CHECK: vpblendmq %ymm21, %ymm23, %ymm21 {%k3} {z}
          vpblendmq %ymm21, %ymm23, %ymm21 {%k3} {z}
// CHECK: vpblendmq (%rcx), %ymm23, %ymm21
          vpblendmq (%rcx), %ymm23, %ymm21
// CHECK: vpblendmq 291(%rax,%r14,8), %ymm23, %ymm21
          vpblendmq 291(%rax,%r14,8), %ymm23, %ymm21
// CHECK: vpblendmq (%rcx){1to4}, %ymm23, %ymm21
          vpblendmq (%rcx){1to4}, %ymm23, %ymm21
// CHECK: vpblendmq 4064(%rdx), %ymm23, %ymm21
          vpblendmq 4064(%rdx), %ymm23, %ymm21
// CHECK: vpblendmq 4096(%rdx), %ymm23, %ymm21
          vpblendmq 4096(%rdx), %ymm23, %ymm21
// CHECK: vpblendmq -4096(%rdx), %ymm23, %ymm21
          vpblendmq -4096(%rdx), %ymm23, %ymm21
// CHECK: vpblendmq -4128(%rdx), %ymm23, %ymm21
          vpblendmq -4128(%rdx), %ymm23, %ymm21
// CHECK: vpblendmq 1016(%rdx){1to4}, %ymm23, %ymm21
          vpblendmq 1016(%rdx){1to4}, %ymm23, %ymm21
// CHECK: vpblendmq 1024(%rdx){1to4}, %ymm23, %ymm21
          vpblendmq 1024(%rdx){1to4}, %ymm23, %ymm21
// CHECK: vpblendmq -1024(%rdx){1to4}, %ymm23, %ymm21
          vpblendmq -1024(%rdx){1to4}, %ymm23, %ymm21
// CHECK: vpblendmq -1032(%rdx){1to4}, %ymm23, %ymm21
          vpblendmq -1032(%rdx){1to4}, %ymm23, %ymm21
