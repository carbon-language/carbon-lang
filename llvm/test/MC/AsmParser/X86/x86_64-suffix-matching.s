// RUN: llvm-mc -triple x86_64 -o - %s | FileCheck %s

// CHECK: addl $0, %eax
        add $0, %eax
// CHECK: addb $255, %al
        add $0xFF, %al
