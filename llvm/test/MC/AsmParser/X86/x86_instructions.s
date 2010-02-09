// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

// CHECK: subb %al, %al
        subb %al, %al

// CHECK: addl $24, %eax
        addl $24, %eax

// CHECK: movl %eax, 10(%ebp)
        movl %eax, 10(%ebp)
// CHECK: movl %eax, 10(%ebp,%ebx)
        movl %eax, 10(%ebp, %ebx)
// CHECK: movl %eax, 10(%ebp,%ebx,4)
        movl %eax, 10(%ebp, %ebx, 4)
// CHECK: movl %eax, 10(,%ebx,4)
        movl %eax, 10(, %ebx, 4)

// CHECK: movl 0, %eax        
        movl 0, %eax
// CHECK: movl $0, %eax        
        movl $0, %eax
        
// CHECK: ret
        ret
        
// FIXME: Check that this matches SUB32ri8
// CHECK: subl $1, %eax
        subl $1, %eax
        
// FIXME: Check that this matches SUB32ri8
// CHECK: subl $-1, %eax
        subl $-1, %eax
        
// FIXME: Check that this matches SUB32ri
// CHECK: subl $256, %eax
        subl $256, %eax

// FIXME: Check that this matches XOR64ri8
// CHECK: xorq $1, %rax
        xorq $1, %rax
        
// FIXME: Check that this matches XOR64ri32
// CHECK: xorq $256, %rax
        xorq $256, %rax

// FIXME: Check that this matches SUB8rr
// CHECK: subb %al, %bl
        subb %al, %bl

// FIXME: Check that this matches SUB16rr
// CHECK: subw %ax, %bx
        subw %ax, %bx
        
// FIXME: Check that this matches SUB32rr
// CHECK: subl %eax, %ebx
        subl %eax, %ebx
        
// FIXME: Check that this matches the correct instruction.
// CHECK: call *%rax
        call *%rax

// FIXME: Check that this matches the correct instruction.
// CHECK: shldl %cl, %eax, %ebx
        shldl %cl, %eax, %ebx

// CHECK: shll $2, %eax
        shll $2, %eax

// CHECK: shll $2, %eax
        sall $2, %eax

// CHECK: rep
// CHECK: insb
        rep;insb

// CHECK: rep
// CHECK: outsb
        rep;outsb

// CHECK: rep
// CHECK: movsb
        rep;movsb

// CHECK: rep
// CHECK: lodsb
        rep;lodsb

// CHECK: rep
// CHECK: stosb
        rep;stosb

// NOTE: repz and repe have the same opcode as rep
// CHECK: rep
// CHECK: cmpsb
        repz;cmpsb

// NOTE: repnz has the same opcode as repne
// CHECK: repne
// CHECK: cmpsb
        repnz;cmpsb

// NOTE: repe and repz have the same opcode as rep
// CHECK: rep
// CHECK: scasb
        repe;scasb

// CHECK: repne
// CHECK: scasb
        repne;scasb

// CHECK: lock
// CHECK: cmpxchgb %al, 0(%ebx)
        lock;cmpxchgb %al, 0(%ebx)

// CHECK: cs
// CHECK: movb 0(%eax), %al
        cs;movb 0(%eax), %al

// CHECK: ss
// CHECK: movb 0(%eax), %al
        ss;movb 0(%eax), %al

// CHECK: ds
// CHECK: movb 0(%eax), %al
        ds;movb 0(%eax), %al

// CHECK: es
// CHECK: movb 0(%eax), %al
        es;movb 0(%eax), %al

// CHECK: fs
// CHECK: movb 0(%eax), %al
        fs;movb 0(%eax), %al

// CHECK: gs
// CHECK: movb 0(%eax), %al
        gs;movb 0(%eax), %al

// CHECK: fadd %st(0)
// CHECK: fadd %st(1)
// CHECK: fadd %st(7)

fadd %st(0)
fadd %st(1)
fadd %st(7)
