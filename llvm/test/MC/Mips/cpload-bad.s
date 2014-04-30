# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 -check-prefix=ASM

        .text
        .option pic2
        .set reorder
        .cpload $25
# ASM: :[[@LINE-1]]:9: warning: .cpload in reorder section
        .set noreorder
        .cpload $32
# ASM: :[[@LINE-1]]:17: error: invalid register
        .cpload $foo
# ASM: :[[@LINE-1]]:17: error: expected register containing function address
        .cpload bar
# ASM: :[[@LINE-1]]:17: error: expected register containing function address
