# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .globl a
# CHECK: .globl b
TEST0:  
        .globl a, b
