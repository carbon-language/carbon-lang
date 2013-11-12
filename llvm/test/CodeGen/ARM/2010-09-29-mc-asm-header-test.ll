; This tests that MC/asm header conversion is smooth and that the
; build attributes are correct

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
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A9-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-A9-HARD
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9-mp | FileCheck %s --check-prefix=CORTEX-A9-MP
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 | FileCheck %s --check-prefix=CORTEX-A15
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 | FileCheck %s --check-prefix=CORTEX-M0
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-M4-SOFT
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-M4-HARD
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5 | FileCheck %s --check-prefix=CORTEX-R5
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a53 | FileCheck %s --check-prefix=CORTEX-A53
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a57 | FileCheck %s --check-prefix=CORTEX-A57

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
; V6M:  .eabi_attribute 7, 77
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
; V7M:  .eabi_attribute 44, 0
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
; CORTEX-A9-NOT:  .eabi_attribute 27
; CORTEX-A9-NOT:  .eabi_attribute 28
; CORTEX-A9-MP:  .eabi_attribute 36, 1
; CORTEX-A9-MP:  .eabi_attribute 42, 1
; CORTEX-A9-MP:  .eabi_attribute 68, 1

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
; CORTEX-M0:  .eabi_attribute 7, 77
; CORTEX-M0:  .eabi_attribute 8, 0
; CORTEX-M0:  .eabi_attribute 9, 1
; CORTEX-M0:  .eabi_attribute 24, 1
; CORTEX-M0:  .eabi_attribute 25, 1
; CORTEX-M0-NOT:  .eabi_attribute 27
; CORTEX-M0-NOT:  .eabi_attribute 28
; CORTEX-M0-NOT:  .eabi_attribute 36
; CORTEX-M0-NOT:  .eabi_attribute 42
; CORTEX-M0-NOT:  .eabi_attribute 68

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
; CORTEX-M4-SOFT:  .eabi_attribute 44, 0
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
; CORTEX-M4-HARD:  .eabi_attribute 44, 0
; CORTEX-M4-HRAD-NOT:  .eabi_attribute 68

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
; CORTEX-A53:  .eabi_attribute 44, 2
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
; CORTEX-A57:  .eabi_attribute 44, 2
; CORTEX-A57:  .eabi_attribute 68, 3

define i32 @f(i64 %z) {
	ret i32 0
}
