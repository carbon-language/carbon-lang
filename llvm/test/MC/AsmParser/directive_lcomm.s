# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .lcomm a,7,4
# CHECK: .lcomm b,8
# CHECK: .lcomm c,0
TEST0:  
        .lcomm a, 8-1, 4
        .lcomm b,8
        .lcomm  c,  0
