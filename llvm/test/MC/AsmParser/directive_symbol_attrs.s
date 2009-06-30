# RUN: llvm-mc %s > %t

# RUN: grep -A 3 TEST0 %t > %t2
# RUN: grep ".globl a" %t2 | count 1
# RUN: grep ".globl b" %t2 | count 1
TEST0:  
        .globl a, b
