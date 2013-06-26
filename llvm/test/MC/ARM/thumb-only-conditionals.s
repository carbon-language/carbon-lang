@ RUN: llvm-mc -triple=thumbv7-apple-ios -o - %s | FileCheck %s

        itte eq
        dmbeq #11
        dsbeq #7
        isbne #15
@ CHECK: itte eq
@ CHECK-NEXT: dmbeq ish
@ CHECK-NEXT: dsbeq nsh
@ CHECK-NEXT: isbne sy

        itet le
        dmble
        dsbgt
        isble
@ CHECK: itet le
@ CHECK-NEXT: dmble sy
@ CHECK-NEXT: dsbgt sy
@ CHECK-NEXT: isble sy
