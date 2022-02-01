// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple arm-gnu-linux-eabi -mcpu cortex-a7 -arm-add-build-attributes %s -o %t.o
// RUN: echo "ENTRY(__entrypoint) SECTIONS { . = 0x10000; .text : { *(.text .text.*) } /DISCARD/ : { *(.ARM.exidx*) *(.gnu.linkonce.armexidx.*) } }" > %t.script
// RUN: ld.lld -T %t.script %t.o -o %t.elf
// RUN: llvm-readobj --sections %t.elf | FileCheck %s

.globl  __entrypoint
__entrypoint:
.fnstart
    bx  lr
 .save {r7, lr}
 .setfp r7, sp, #0
 .fnend
// Check that .ARM.exidx/.gnu.linkonce.armexidx
// are correctly removed if they were added.
// CHECK-NOT: .ARM.exidx
// CHECK-NOT: .gnu.linkonce.armexidx.
