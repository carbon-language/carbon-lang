// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# Simple instructions
        subb %al, %al
# CHECK: MCInst(opcode=1831, operands=[reg:2, reg:0, reg:2])
