// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

.irp reg,%eax,%ebx
        pushl \reg
.endr

// CHECK: pushl %eax
// CHECK: pushl %ebx
