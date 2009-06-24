# RUN: llvm-mc %s > %t

# RUN: grep -A 1 TEST0 %t > %t2
# RUN: not grep ".byte" %t2
TEST0:  
        .ascii

# RUN: grep -A 1 TEST1 %t > %t2
# RUN: not grep "byte" %t2
TEST1:  
        .asciz

# RUN: grep -A 2 TEST2 %t > %t2
# RUN: grep ".byte 65" %t2 | count 1
TEST2:  
        .ascii "A"

# RUN: grep -A 5 TEST3 %t > %t2
# RUN: grep ".byte 66" %t2 | count 1
# RUN: grep ".byte 67" %t2 | count 1
# RUN: grep ".byte 0" %t2 | count 2
TEST3:  
        .asciz "B", "C"

       