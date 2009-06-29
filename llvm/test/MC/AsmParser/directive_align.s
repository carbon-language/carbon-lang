# RUN: llvm-mc %s > %t

# RUN: grep -A 2 TEST0 %t > %t2
# RUN: grep ".p2align 1, 0" %t2 | count 1
TEST0:  
        .align 1

# RUN: grep -A 2 TEST1 %t > %t2
# RUN: grep ".p2alignl 3, 0, 2" %t2 | count 1
TEST1:  
        .align32 3,,2

# RUN: grep -A 2 TEST2 %t > %t2
# RUN: grep ".balign 3, 10" %t2 | count 1
TEST2:  
        .balign 3,10
