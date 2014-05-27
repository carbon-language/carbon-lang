; This tests that MC/asm header conversion is smooth and that the
; build attributes are correct

; RUN: llc < %s -mtriple=thumbv5-linux-gnueabi -mcpu=xscale | FileCheck %s --check-prefix=XSCALE
; RUN: llc < %s -mtriple=armv6-linux-gnueabi | FileCheck %s --check-prefix=V6
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi | FileCheck %s --check-prefix=V6M
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mcpu=arm1156t2f-s | FileCheck %s --check-prefix=ARM1156T2F-S
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi | FileCheck %s --check-prefix=V7M
; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi | FileCheck %s --check-prefix=Vt8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-neon,-crypto | FileCheck %s --check-prefix=V8-FPARMv8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-fp-armv8,-crypto | FileCheck %s --check-prefix=V8-NEON
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-crypto | FileCheck %s --check-prefix=V8-FPARMv8-NEON
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8-FPARMv8-NEON-CRYPTO
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 | FileCheck %s --check-prefix=CORTEX-A5-DEFAULT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -mattr=-neon,+d16 | FileCheck %s --check-prefix=CORTEX-A5-NONEON
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-A5-NOFPU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A9-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-A9-HARD
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 | FileCheck %s --check-prefix=CORTEX-A12-DEFAULT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-A12-NOFPU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9-mp | FileCheck %s --check-prefix=CORTEX-A9-MP
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 | FileCheck %s --check-prefix=CORTEX-A15
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 | FileCheck %s --check-prefix=CORTEX-M0
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m3 | FileCheck %s --check-prefix=CORTEX-M3
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-M4-SOFT
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-M4-HARD
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5 | FileCheck %s --check-prefix=CORTEX-R5
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a53 | FileCheck %s --check-prefix=CORTEX-A53
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a57 | FileCheck %s --check-prefix=CORTEX-A57
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 | FileCheck %s  --check-prefix=CORTEX-A7-CHECK
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=-vfp2,-vfp3,-vfp4,-neon | FileCheck %s --check-prefix=CORTEX-A7-NOFPU
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+vfp4,-neon | FileCheck %s --check-prefix=CORTEX-A7-FPUV4
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+vfp4,,+d16,-neon | FileCheck %s --check-prefix=CORTEX-A7-FPUV4
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -relocation-model=pic | FileCheck %s --check-prefix=RELOC-PIC
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -relocation-model=static | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -relocation-model=default | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -relocation-model=dynamic-no-pic | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi | FileCheck %s --check-prefix=RELOC-OTHER

; XSCALE:      .eabi_attribute 6, 5
; XSCALE:      .eabi_attribute 8, 1
; XSCALE:      .eabi_attribute 9, 1

; V6:   .eabi_attribute 6, 6
; V6:   .eabi_attribute 8, 1
; V6:   .eabi_attribute 24, 1
; V6:   .eabi_attribute 25, 1
; V6-NOT:   .eabi_attribute 27
; V6-NOT:   .eabi_attribute 28
; V6-NOT:    .eabi_attribute 36
; V6-NOT:    .eabi_attribute 42
; V6-NOT:    .eabi_attribute 68

; V6M:  .eabi_attribute 6, 12
; V6M-NOT:  .eabi_attribute 7
; V6M:  .eabi_attribute 8, 0
; V6M:  .eabi_attribute 9, 1
; V6M:  .eabi_attribute 24, 1
; V6M:  .eabi_attribute 25, 1
; V6M-NOT:  .eabi_attribute 27
; V6M-NOT:  .eabi_attribute 28
; V6M-NOT:  .eabi_attribute 36
; V6M-NOT:  .eabi_attribute 42
; V6M-NOT:  .eabi_attribute 68

; ARM1156T2F-S: .cpu arm1156t2f-s
; ARM1156T2F-S: .eabi_attribute 6, 8
; ARM1156T2F-S: .eabi_attribute 8, 1
; ARM1156T2F-S: .eabi_attribute 9, 2
; ARM1156T2F-S: .fpu vfpv2
; ARM1156T2F-S: .eabi_attribute 20, 1
; ARM1156T2F-S: .eabi_attribute 21, 1
; ARM1156T2F-S: .eabi_attribute 23, 3
; ARM1156T2F-S: .eabi_attribute 24, 1
; ARM1156T2F-S: .eabi_attribute 25, 1
; ARM1156T2F-S-NOT: .eabi_attribute 27
; ARM1156T2F-S-NOT: .eabi_attribute 28
; ARM1156T2F-S-NOT: .eabi_attribute 36
; ARM1156T2F-S-NOT:    .eabi_attribute 42
; ARM1156T2F-S-NOT:    .eabi_attribute 68

; V7M:  .eabi_attribute 6, 10
; V7M:  .eabi_attribute 7, 77
; V7M:  .eabi_attribute 8, 0
; V7M:  .eabi_attribute 9, 2
; V7M:  .eabi_attribute 24, 1
; V7M:  .eabi_attribute 25, 1
; V7M-NOT:  .eabi_attribute 27
; V7M-NOT:  .eabi_attribute 28
; V7M-NOT:  .eabi_attribute 36
; V7M-NOT:  .eabi_attribute 42
; V7M-NOT:  .eabi_attribute 44
; V7M-NOT:  .eabi_attribute 68

; V7:      .syntax unified
; V7: .eabi_attribute 6, 10
; V7: .eabi_attribute 20, 1
; V7: .eabi_attribute 21, 1
; V7: .eabi_attribute 23, 3
; V7: .eabi_attribute 24, 1
; V7: .eabi_attribute 25, 1
; V7-NOT: .eabi_attribute 27
; V7-NOT: .eabi_attribute 28
; V7-NOT: .eabi_attribute 36
; V7-NOT:    .eabi_attribute 42
; V7-NOT:    .eabi_attribute 68

; V8:      .syntax unified
; V8: .eabi_attribute 6, 14

; Vt8:     .syntax unified
; Vt8: .eabi_attribute 6, 14

; V8-FPARMv8:      .syntax unified
; V8-FPARMv8: .eabi_attribute 6, 14
; V8-FPARMv8: .fpu fp-armv8

; V8-NEON:      .syntax unified
; V8-NEON: .eabi_attribute 6, 14
; V8-NEON: .fpu neon
; V8-NEON: .eabi_attribute 12, 3

; V8-FPARMv8-NEON:      .syntax unified
; V8-FPARMv8-NEON: .eabi_attribute 6, 14
; V8-FPARMv8-NEON: .fpu neon-fp-armv8
; V8-FPARMv8-NEON: .eabi_attribute 12, 3

; V8-FPARMv8-NEON-CRYPTO:      .syntax unified
; V8-FPARMv8-NEON-CRYPTO: .eabi_attribute 6, 14
; V8-FPARMv8-NEON-CRYPTO: .fpu crypto-neon-fp-armv8
; V8-FPARMv8-NEON-CRYPTO: .eabi_attribute 12, 3

; Tag_CPU_arch	'ARMv7'
; CORTEX-A7-CHECK: .eabi_attribute	6, 10
; CORTEX-A7-NOFPU: .eabi_attribute	6, 10
; CORTEX-A7-FPUV4: .eabi_attribute	6, 10

; Tag_CPU_arch_profile 'A'
; CORTEX-A7-CHECK: .eabi_attribute	7, 65
; CORTEX-A7-NOFPU: .eabi_attribute	7, 65
; CORTEX-A7-FPUV4: .eabi_attribute	7, 65

; Tag_ARM_ISA_use
; CORTEX-A7-CHECK: .eabi_attribute	8, 1
; CORTEX-A7-NOFPU: .eabi_attribute	8, 1
; CORTEX-A7-FPUV4: .eabi_attribute	8, 1

; Tag_THUMB_ISA_use
; CORTEX-A7-CHECK: .eabi_attribute	9, 2
; CORTEX-A7-NOFPU: .eabi_attribute	9, 2
; CORTEX-A7-FPUV4: .eabi_attribute	9, 2

; CORTEX-A7-CHECK: .fpu	neon-vfpv4
; CORTEX-A7-NOFPU-NOT: .fpu
; CORTEX-A7-FPUV4: .fpu	vfpv4

; Tag_ABI_FP_denormal
; CORTEX-A7-CHECK: .eabi_attribute	20, 1
; CORTEX-A7-NOFPU: .eabi_attribute	20, 1
; CORTEX-A7-FPUV4: .eabi_attribute	20, 1

; Tag_ABI_FP_exceptions
; CORTEX-A7-CHECK: .eabi_attribute	21, 1
; CORTEX-A7-NOFPU: .eabi_attribute	21, 1
; CORTEX-A7-FPUV4: .eabi_attribute	21, 1

; Tag_ABI_FP_number_model
; CORTEX-A7-CHECK: .eabi_attribute	23, 3
; CORTEX-A7-NOFPU: .eabi_attribute	23, 3
; CORTEX-A7-FPUV4: .eabi_attribute	23, 3

; Tag_ABI_align_needed
; CORTEX-A7-CHECK: .eabi_attribute	24, 1
; CORTEX-A7-NOFPU: .eabi_attribute	24, 1
; CORTEX-A7-FPUV4: .eabi_attribute	24, 1

; Tag_ABI_align_preserved
; CORTEX-A7-CHECK: .eabi_attribute	25, 1
; CORTEX-A7-NOFPU: .eabi_attribute	25, 1
; CORTEX-A7-FPUV4: .eabi_attribute	25, 1

; Tag_FP_HP_extension
; CORTEX-A7-CHECK: .eabi_attribute	36, 1
; CORTEX-A7-NOFPU: .eabi_attribute	36, 1
; CORTEX-A7-FPUV4: .eabi_attribute	36, 1

; Tag_MPextension_use
; CORTEX-A7-CHECK: .eabi_attribute	42, 1
; CORTEX-A7-NOFPU: .eabi_attribute	42, 1
; CORTEX-A7-FPUV4: .eabi_attribute	42, 1

; Tag_DIV_use
; CORTEX-A7-CHECK: .eabi_attribute	44, 2
; CORTEX-A7-NOFPU: .eabi_attribute	44, 2
; CORTEX-A7-FPUV4: .eabi_attribute	44, 2

; Tag_Virtualization_use
; CORTEX-A7-CHECK: .eabi_attribute	68, 3
; CORTEX-A7-NOFPU: .eabi_attribute	68, 3
; CORTEX-A7-FPUV4: .eabi_attribute	68, 3

; CORTEX-A5-DEFAULT:        .cpu    cortex-a5
; CORTEX-A5-DEFAULT:        .eabi_attribute 6, 10
; CORTEX-A5-DEFAULT:        .eabi_attribute 7, 65
; CORTEX-A5-DEFAULT:        .eabi_attribute 8, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 9, 2
; CORTEX-A5-DEFAULT:        .fpu    neon-vfpv4
; CORTEX-A5-DEFAULT:        .eabi_attribute 20, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 21, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 23, 3
; CORTEX-A5-DEFAULT:        .eabi_attribute 24, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 25, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 42, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 68, 1

; CORTEX-A5-NONEON:        .cpu    cortex-a5
; CORTEX-A5-NONEON:        .eabi_attribute 6, 10
; CORTEX-A5-NONEON:        .eabi_attribute 7, 65
; CORTEX-A5-NONEON:        .eabi_attribute 8, 1
; CORTEX-A5-NONEON:        .eabi_attribute 9, 2
; CORTEX-A5-NONEON:        .fpu    vfpv4-d16
; CORTEX-A5-NONEON:        .eabi_attribute 20, 1
; CORTEX-A5-NONEON:        .eabi_attribute 21, 1
; CORTEX-A5-NONEON:        .eabi_attribute 23, 3
; CORTEX-A5-NONEON:        .eabi_attribute 24, 1
; CORTEX-A5-NONEON:        .eabi_attribute 25, 1
; CORTEX-A5-NONEON:        .eabi_attribute 42, 1
; CORTEX-A5-NONEON:        .eabi_attribute 68, 1

; CORTEX-A5-NOFPU:        .cpu    cortex-a5
; CORTEX-A5-NOFPU:        .eabi_attribute 6, 10
; CORTEX-A5-NOFPU:        .eabi_attribute 7, 65
; CORTEX-A5-NOFPU:        .eabi_attribute 8, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 9, 2
; CORTEX-A5-NOFPU-NOT:    .fpu
; CORTEX-A5-NOFPU:        .eabi_attribute 20, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 21, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 23, 3
; CORTEX-A5-NOFPU:        .eabi_attribute 24, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 25, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 42, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 68, 1

; CORTEX-A9-SOFT:  .cpu cortex-a9
; CORTEX-A9-SOFT:  .eabi_attribute 6, 10
; CORTEX-A9-SOFT:  .eabi_attribute 7, 65
; CORTEX-A9-SOFT:  .eabi_attribute 8, 1
; CORTEX-A9-SOFT:  .eabi_attribute 9, 2
; CORTEX-A9-SOFT:  .fpu neon
; CORTEX-A9-SOFT:  .eabi_attribute 20, 1
; CORTEX-A9-SOFT:  .eabi_attribute 21, 1
; CORTEX-A9-SOFT:  .eabi_attribute 23, 3
; CORTEX-A9-SOFT:  .eabi_attribute 24, 1
; CORTEX-A9-SOFT:  .eabi_attribute 25, 1
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 27
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 28
; CORTEX-A9-SOFT:  .eabi_attribute 36, 1
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 42
; CORTEX-A9-SOFT:  .eabi_attribute 68, 1

; CORTEX-A9-HARD:  .cpu cortex-a9
; CORTEX-A9-HARD:  .eabi_attribute 6, 10
; CORTEX-A9-HARD:  .eabi_attribute 7, 65
; CORTEX-A9-HARD:  .eabi_attribute 8, 1
; CORTEX-A9-HARD:  .eabi_attribute 9, 2
; CORTEX-A9-HARD:  .fpu neon
; CORTEX-A9-HARD:  .eabi_attribute 20, 1
; CORTEX-A9-HARD:  .eabi_attribute 21, 1
; CORTEX-A9-HARD:  .eabi_attribute 23, 3
; CORTEX-A9-HARD:  .eabi_attribute 24, 1
; CORTEX-A9-HARD:  .eabi_attribute 25, 1
; CORTEX-A9-HARD-NOT:  .eabi_attribute 27
; CORTEX-A9-HARD:  .eabi_attribute 28, 1
; CORTEX-A9-HARD:  .eabi_attribute 36, 1
; CORTEX-A9-HARD-NOT:  .eabi_attribute 42
; CORTEX-A9-HARD:  .eabi_attribute 68, 1

; CORTEX-A9-MP:  .cpu cortex-a9-mp
; CORTEX-A9-MP:  .eabi_attribute 6, 10
; CORTEX-A9-MP:  .eabi_attribute 7, 65
; CORTEX-A9-MP:  .eabi_attribute 8, 1
; CORTEX-A9-MP:  .eabi_attribute 9, 2
; CORTEX-A9-MP:  .fpu neon
; CORTEX-A9-MP:  .eabi_attribute 20, 1
; CORTEX-A9-MP:  .eabi_attribute 21, 1
; CORTEX-A9-MP:  .eabi_attribute 23, 3
; CORTEX-A9-MP:  .eabi_attribute 24, 1
; CORTEX-A9-MP:  .eabi_attribute 25, 1
; CORTEX-A9-MP-NOT:  .eabi_attribute 27
; CORTEX-A9-MP-NOT:  .eabi_attribute 28
; CORTEX-A9-MP:  .eabi_attribute 36, 1
; CORTEX-A9-MP:  .eabi_attribute 42, 1
; CORTEX-A9-MP:  .eabi_attribute 68, 1

; CORTEX-A12-DEFAULT:  .cpu cortex-a12
; CORTEX-A12-DEFAULT:  .eabi_attribute 6, 10
; CORTEX-A12-DEFAULT:  .eabi_attribute 7, 65
; CORTEX-A12-DEFAULT:  .eabi_attribute 8, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 9, 2
; CORTEX-A12-DEFAULT:  .fpu neon-vfpv4
; CORTEX-A12-DEFAULT:  .eabi_attribute 20, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 21, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 23, 3
; CORTEX-A12-DEFAULT:  .eabi_attribute 24, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 25, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 42, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 44, 2
; CORTEX-A12-DEFAULT:  .eabi_attribute 68, 3

; CORTEX-A12-NOFPU:  .cpu cortex-a12
; CORTEX-A12-NOFPU:  .eabi_attribute 6, 10
; CORTEX-A12-NOFPU:  .eabi_attribute 7, 65
; CORTEX-A12-NOFPU:  .eabi_attribute 8, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 9, 2
; CORTEX-A12-NOFPU-NOT:  .fpu
; CORTEX-A12-NOFPU:  .eabi_attribute 20, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 21, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 23, 3
; CORTEX-A12-NOFPU:  .eabi_attribute 24, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 25, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 42, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 44, 2
; CORTEX-A12-NOFPU:  .eabi_attribute 68, 3

; CORTEX-A15: .cpu cortex-a15
; CORTEX-A15: .eabi_attribute 6, 10
; CORTEX-A15: .eabi_attribute 7, 65
; CORTEX-A15: .eabi_attribute 8, 1
; CORTEX-A15: .eabi_attribute 9, 2
; CORTEX-A15: .fpu neon-vfpv4
; CORTEX-A15: .eabi_attribute 20, 1
; CORTEX-A15: .eabi_attribute 21, 1
; CORTEX-A15: .eabi_attribute 23, 3
; CORTEX-A15: .eabi_attribute 24, 1
; CORTEX-A15: .eabi_attribute 25, 1
; CORTEX-A15-NOT: .eabi_attribute 27
; CORTEX-A15-NOT: .eabi_attribute 28
; CORTEX-A15: .eabi_attribute 36, 1
; CORTEX-A15: .eabi_attribute 42, 1
; CORTEX-A15: .eabi_attribute 44, 2
; CORTEX-A15: .eabi_attribute 68, 3

; CORTEX-M0:  .cpu cortex-m0
; CORTEX-M0:  .eabi_attribute 6, 12
; CORTEX-M0-NOT:  .eabi_attribute 7
; CORTEX-M0:  .eabi_attribute 8, 0
; CORTEX-M0:  .eabi_attribute 9, 1
; CORTEX-M0:  .eabi_attribute 24, 1
; CORTEX-M0:  .eabi_attribute 25, 1
; CORTEX-M0-NOT:  .eabi_attribute 27
; CORTEX-M0-NOT:  .eabi_attribute 28
; CORTEX-M0-NOT:  .eabi_attribute 36
; CORTEX-M0-NOT:  .eabi_attribute 42
; CORTEX-M0-NOT:  .eabi_attribute 68

; CORTEX-M3:  .cpu cortex-m3
; CORTEX-M3:  .eabi_attribute 6, 10
; CORTEX-M3:  .eabi_attribute 7, 77
; CORTEX-M3:  .eabi_attribute 8, 0
; CORTEX-M3:  .eabi_attribute 9, 2
; CORTEX-M3:  .eabi_attribute 20, 1
; CORTEX-M3:  .eabi_attribute 21, 1
; CORTEX-M3:  .eabi_attribute 23, 3
; CORTEX-M3:  .eabi_attribute 24, 1
; CORTEX-M3:  .eabi_attribute 25, 1
; CORTEX-M3-NOT:  .eabi_attribute 27
; CORTEX-M3-NOT:  .eabi_attribute 28
; CORTEX-M3-NOT:  .eabi_attribute 36
; CORTEX-M3-NOT:  .eabi_attribute 42
; CORTEX-M3-NOT:  .eabi_attribute 44
; CORTEX-M3-NOT:  .eabi_attribute 68

; CORTEX-M4-SOFT:  .cpu cortex-m4
; CORTEX-M4-SOFT:  .eabi_attribute 6, 13
; CORTEX-M4-SOFT:  .eabi_attribute 7, 77
; CORTEX-M4-SOFT:  .eabi_attribute 8, 0
; CORTEX-M4-SOFT:  .eabi_attribute 9, 2
; CORTEX-M4-SOFT:  .fpu vfpv4-d16
; CORTEX-M4-SOFT:  .eabi_attribute 20, 1
; CORTEX-M4-SOFT:  .eabi_attribute 21, 1
; CORTEX-M4-SOFT:  .eabi_attribute 23, 3
; CORTEX-M4-SOFT:  .eabi_attribute 24, 1
; CORTEX-M4-SOFT:  .eabi_attribute 25, 1
; CORTEX-M4-SOFT:  .eabi_attribute 27, 1
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 28
; CORTEX-M4-SOFT:  .eabi_attribute 36, 1
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 42
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 44
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 68

; CORTEX-M4-HARD:  .cpu cortex-m4
; CORTEX-M4-HARD:  .eabi_attribute 6, 13
; CORTEX-M4-HARD:  .eabi_attribute 7, 77
; CORTEX-M4-HARD:  .eabi_attribute 8, 0
; CORTEX-M4-HARD:  .eabi_attribute 9, 2
; CORTEX-M4-HARD:  .fpu vfpv4-d16
; CORTEX-M4-HARD:  .eabi_attribute 20, 1
; CORTEX-M4-HARD:  .eabi_attribute 21, 1
; CORTEX-M4-HARD:  .eabi_attribute 23, 3
; CORTEX-M4-HARD:  .eabi_attribute 24, 1
; CORTEX-M4-HARD:  .eabi_attribute 25, 1
; CORTEX-M4-HARD:  .eabi_attribute 27, 1
; CORTEX-M4-HARD:  .eabi_attribute 28, 1
; CORTEX-M4-HARD:  .eabi_attribute 36, 1
; CORTEX-M4-HARD-NOT:  .eabi_attribute 42
; CORTEX-M4-HARD-NOT:  .eabi_attribute 44
; CORTEX-M4-HARD-NOT:  .eabi_attribute 68

; CORTEX-R5:  .cpu cortex-r5
; CORTEX-R5:  .eabi_attribute 6, 10
; CORTEX-R5:  .eabi_attribute 7, 82
; CORTEX-R5:  .eabi_attribute 8, 1
; CORTEX-R5:  .eabi_attribute 9, 2
; CORTEX-R5:  .fpu vfpv3-d16
; CORTEX-R5:  .eabi_attribute 20, 1
; CORTEX-R5:  .eabi_attribute 21, 1
; CORTEX-R5:  .eabi_attribute 23, 3
; CORTEX-R5:  .eabi_attribute 24, 1
; CORTEX-R5:  .eabi_attribute 25, 1
; CORTEX-R5:  .eabi_attribute 27, 1
; CORTEX-R5-NOT:  .eabi_attribute 28
; CORTEX-R5-NOT:  .eabi_attribute 36
; CORTEX-R5-NOT:  .eabi_attribute 42
; CORTEX-R5:  .eabi_attribute 44, 2
; CORTEX-R5-NOT:  .eabi_attribute 68

; CORTEX-A53:  .cpu cortex-a53
; CORTEX-A53:  .eabi_attribute 6, 14
; CORTEX-A53:  .eabi_attribute 7, 65
; CORTEX-A53:  .eabi_attribute 8, 1
; CORTEX-A53:  .eabi_attribute 9, 2
; CORTEX-A53:  .fpu crypto-neon-fp-armv8
; CORTEX-A53:  .eabi_attribute 12, 3
; CORTEX-A53:  .eabi_attribute 24, 1
; CORTEX-A53:  .eabi_attribute 25, 1
; CORTEX-A53-NOT:  .eabi_attribute 27
; CORTEX-A53-NOT:  .eabi_attribute 28
; CORTEX-A53:  .eabi_attribute 36, 1
; CORTEX-A53:  .eabi_attribute 42, 1
; CORTEX-A53-NOT:  .eabi_attribute 44
; CORTEX-A53:  .eabi_attribute 68, 3

; CORTEX-A57:  .cpu cortex-a57
; CORTEX-A57:  .eabi_attribute 6, 14
; CORTEX-A57:  .eabi_attribute 7, 65
; CORTEX-A57:  .eabi_attribute 8, 1
; CORTEX-A57:  .eabi_attribute 9, 2
; CORTEX-A57:  .fpu crypto-neon-fp-armv8
; CORTEX-A57:  .eabi_attribute 12, 3
; CORTEX-A57:  .eabi_attribute 24, 1
; CORTEX-A57:  .eabi_attribute 25, 1
; CORTEX-A57-NOT:  .eabi_attribute 27
; CORTEX-A57-NOT:  .eabi_attribute 28
; CORTEX-A57:  .eabi_attribute 36, 1
; CORTEX-A57:  .eabi_attribute 42, 1
; CORTEX-A57-NOT:  .eabi_attribute 44
; CORTEX-A57:  .eabi_attribute 68, 3

; RELOC-PIC:  .eabi_attribute 15, 1
; RELOC-PIC:  .eabi_attribute 16, 1
; RELOC-PIC:  .eabi_attribute 17, 2
; RELOC-OTHER:  .eabi_attribute 17, 1

define i32 @f(i64 %z) {
	ret i32 0
}
