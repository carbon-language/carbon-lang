# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".byte 0" %t2 | count 1
TEST0:  
        .space 1

# RUN: grep -A 3 TEST1 %t > %t2
# RUN: grep ".byte 3" %t2 | count 2
TEST1:  
        .space 2, 3
