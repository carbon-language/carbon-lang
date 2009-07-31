// FIXME: Switch back to FileCheck once we print actual instructions

// RUN: llvm-mc -triple i386-unknown-unknown %s > %t

# Simple instructions
        subb %al, %al
// RUN: grep {MCInst(opcode=.*, operands=.reg:2, reg:0, reg:2.)} %t
