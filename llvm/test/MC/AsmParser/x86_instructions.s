// FIXME: Switch back to FileCheck once we print actual instructions

// FIXME: Disabled until the generated code stops crashing gcc 4.0.        
// XFAIL: *
        
// RUN: llvm-mc -triple i386-unknown-unknown %s > %t

// RUN: grep {MCInst(opcode=.*, operands=.reg:2, reg:0, reg:2.)} %t
        subb %al, %al

// RUN: grep {MCInst(opcode=.*, operands=.reg:19, reg:0, val:24.)} %t
        addl $24, %eax

// RUN: grep {MCInst(opcode=.*, operands=.reg:20, imm:1, reg:0, val:10, reg:0, reg:19.)} %t
        movl %eax, 10(%ebp)
// RUN: grep {MCInst(opcode=.*, operands=.reg:20, imm:1, reg:21, val:10, reg:0, reg:19.)} %t
        movl %eax, 10(%ebp, %ebx)
// RUN: grep {MCInst(opcode=.*, operands=.reg:20, imm:4, reg:21, val:10, reg:0, reg:19.)} %t
        movl %eax, 10(%ebp, %ebx, 4)
// RUN: grep {MCInst(opcode=.*, operands=.reg:0, imm:4, reg:21, val:10, reg:0, reg:19.)} %t
        movl %eax, 10(, %ebx, 4)
