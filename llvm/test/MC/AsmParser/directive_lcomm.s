# RUN: llvm-mc -triple i386-apple-darwin10 %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .zerofill __DATA,__bss,a,7,4
# CHECK: .zerofill __DATA,__bss,b,8
# CHECK: .zerofill __DATA,__bss,c,0
TEST0:  
        .lcomm a, 8-1, 4
        .lcomm b,8
        .lcomm  c,  0
