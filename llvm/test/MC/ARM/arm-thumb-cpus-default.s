@ RUN: llvm-mc -show-encoding -arch=arm < %s | FileCheck %s --check-prefix=CHECK-ARM-ONLY
@ RUN: llvm-mc -show-encoding -triple=armv4t < %s | FileCheck %s --check-prefix=CHECK-ARM-THUMB
@ RUN: llvm-mc -show-encoding -arch=arm -mcpu=cortex-a15 < %s| FileCheck %s --check-prefix=CHECK-ARM-THUMB
@ RUN: llvm-mc -show-encoding -arch=arm -mcpu=cortex-m3 < %s | FileCheck %s --check-prefix=CHECK-THUMB-ONLY
@ RUN: llvm-mc -show-encoding -triple=armv7m < %s | FileCheck %s --check-prefix=CHECK-THUMB-ONLY
@ RUN: llvm-mc -show-encoding -triple=armv6m < %s | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

        @ Make sure the architecture chosen by LLVM defaults to a compatible
        @ ARM/Thumb mode.
        movs r0, r0
@ CHECK-ARM-THUMB: movs r0, r0 @ encoding: [0x00,0x00,0xb0,0xe1]
@ CHECK-ARM-ONLY: movs r0, r0 @ encoding: [0x00,0x00,0xb0,0xe1]
@ CHECK-THUMB-ONLY: movs r0, r0 @ encoding: [0x00,0x00]
