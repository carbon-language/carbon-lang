// FIXME: Switch back to FileCheck once we print actual instructions
        
// RUN: llvm-mc -triple x86_64-unknown-unknown %s > %t

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

// FIXME: Check that this matches SUB32ri8
// RUN: grep {MCInst(opcode=.*, operands=.reg:19, reg:0, val:1.)} %t
        subl $1, %eax
        
// FIXME: Check that this matches SUB32ri8
// RUN: grep {MCInst(opcode=.*, operands=.reg:19, reg:0, val:-1.)} %t
        subl $-1, %eax
        
// FIXME: Check that this matches SUB32ri
// RUN: grep {MCInst(opcode=.*, operands=.reg:19, reg:0, val:256.)} %t
        subl $256, %eax

// FIXME: Check that this matches XOR64ri8
// RUN: grep {MCInst(opcode=.*, operands=.reg:80, reg:0, val:1.)} %t
        xorq $1, %rax
        
// FIXME: Check that this matches XOR64ri32
// RUN: grep {MCInst(opcode=.*, operands=.reg:80, reg:0, val:256.)} %t
        xorq $256, %rax

// FIXME: Check that this matches SUB8rr
// RUN: grep {MCInst(opcode=.*, operands=.reg:5, reg:0, reg:2.)} %t
        subb %al, %bl

// FIXME: Check that this matches SUB16rr
// RUN: grep {MCInst(opcode=.*, operands=.reg:8, reg:0, reg:3.)} %t
        subw %ax, %bx
        
// FIXME: Check that this matches SUB32rr
// RUN: grep {MCInst(opcode=.*, operands=.reg:21, reg:0, reg:19.)} %t
        subl %eax, %ebx
        
// FIXME: Check that this matches the correct instruction.
// RUN: grep {MCInst(opcode=.*, operands=.reg:80.)} %t
        call *%rax
