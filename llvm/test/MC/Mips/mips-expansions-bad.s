# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1

        .text
        li $5, 0x100000000 # CHECK: :[[@LINE]]:9: error: instruction requires a 64-bit architecture
        dli $5, 1 # CHECK: :[[@LINE]]:9: error: instruction requires a 64-bit architecture
