# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".set a, 0" %t2
TEST0:  
        .set a, 0
        