# RUN: llvm-mc %s > %t

# RUN: grep -A 3 TEST0 %t > %t2
# RUN: grep ".comm a,6,2" %t2 | count 1
# RUN: grep ".comm b,8" %t2 | count 1
TEST0:  
        .comm a, 4+2, 2
        .comm b,8
