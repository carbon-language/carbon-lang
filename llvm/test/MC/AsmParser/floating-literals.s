# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: .long	1067412619
# CHECK: .long	1075000115
# CHECK: .long	1077936128
# CHECK: .long	1082549862
.single 1.2455, +2.3, 3, + 4.2
        
# CHECK: .quad	4617315517961601024
# CHECK: .quad	4597526701198935065
# CHECK: .quad	-4600933674317040845
.double 5, .232, -11.1
