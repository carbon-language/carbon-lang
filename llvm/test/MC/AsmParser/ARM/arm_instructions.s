@ RUN: llvm-mc -triple arm-unknown-unknown %s | FileCheck %s

@ CHECK: nop
        nop

@ CHECK: nopeq
        nopeq

