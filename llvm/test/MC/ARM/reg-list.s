@ RUN: llvm-mc -triple thumb-apple-darwin10 -show-encoding < %s 2> %t | FileCheck %s
@ RUN: FileCheck --check-prefix=CHECK-WARNINGS < %t %s

        push    {r7, lr}
@ CHECK-WARNINGS: register not in ascending order in register list

        push	{lr, r7}
@ CHECK: push {lr, r7}
