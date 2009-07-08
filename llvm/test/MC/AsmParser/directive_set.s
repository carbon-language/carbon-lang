# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .set a, 0
TEST0:  
        .set a, 0
        