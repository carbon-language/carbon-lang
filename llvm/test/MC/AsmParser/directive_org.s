# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".org 1, 0" %t2 | count 1
TEST0:  
        .org 1

# RUN: grep -A 2 TEST1 %t > %t2
# RUN: grep ".org 1, 3" %t2 | count 1
TEST1:  
        .org 1, 3
