# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".byte 0" %t2 | count 1
TEST0:  
        .byte 0

# RUN: grep -A 2 TEST1 %t > %t2
# RUN: grep ".short 3" %t2 | count 1
TEST1:  
        .short 3

# RUN: grep -A 2 TEST2 %t > %t2
# RUN: grep ".long 8" %t2 | count 1
TEST2:  
        .long 8

# RUN: grep -A 2 TEST3 %t > %t2
# RUN: grep ".quad 9" %t2 | count 1
TEST3:  
        .quad 9
