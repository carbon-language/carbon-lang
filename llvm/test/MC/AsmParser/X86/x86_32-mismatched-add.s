// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s
// XFAIL: *

// CHECK: addl	$4294967295, %eax       # encoding: [0x83,0xc0,0xff]
        addl $0xFFFFFFFF, %eax

// CHECK: addl	$65535, %eax       # encoding: [0x66,0x83,0xc0,0xff]
        addw $0xFFFF, %ax
