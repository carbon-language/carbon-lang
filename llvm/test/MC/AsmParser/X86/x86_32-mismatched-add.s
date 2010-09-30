// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: addl	$4294967295, %eax       # encoding: [0x83,0xc0,0xff]
        addl $0xFFFFFFFF, %eax

// CHECK: addw	$65535, %ax       # encoding: [0x66,0x83,0xc0,0xff]
        addw $0xFFFF, %ax
