# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".byte 10" %t2 | count 1
TEST0:  
        .fill 1, 1, 10

# RUN: grep -A 3 TEST1 %t > %t2
# RUN: grep ".short 3" %t2 | count 2
TEST1:  
        .fill 2, 2, 3
