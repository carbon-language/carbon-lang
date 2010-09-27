// RUN: llvm-mc -triple x86_64-unknown-unknown -show-encoding %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

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
// CHECK: callq *%rax
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


// rdar://8470918
smovb // CHECK: movsb
smovw // CHECK: movsw
smovl // CHECK: movsl
smovq // CHECK: movsq

// rdar://8456361
// CHECK: rep
// CHECK: movsl
        rep movsd

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
// CHECK: cmpxchgb %al, (%ebx)
        lock;cmpxchgb %al, 0(%ebx)

// CHECK: cs
// CHECK: movb (%eax), %al
        cs;movb 0(%eax), %al

// CHECK: ss
// CHECK: movb (%eax), %al
        ss;movb 0(%eax), %al

// CHECK: ds
// CHECK: movb (%eax), %al
        ds;movb 0(%eax), %al

// CHECK: es
// CHECK: movb (%eax), %al
        es;movb 0(%eax), %al

// CHECK: fs
// CHECK: movb (%eax), %al
        fs;movb 0(%eax), %al

// CHECK: gs
// CHECK: movb (%eax), %al
        gs;movb 0(%eax), %al

// CHECK: fadd %st(0)
// CHECK: fadd %st(1)
// CHECK: fadd %st(7)

fadd %st(0)
fadd %st(1)
fadd %st(7)

// CHECK: leal 0, %eax
        leal 0, %eax

// rdar://7986634 - Insensitivity on opcodes.
// CHECK: int3
INT3


// Allow scale factor without index register.
// CHECK: movaps	%xmm3, (%esi)
// CHECK-STDERR: warning: scale factor without index register is ignored
movaps %xmm3, (%esi, 2)

// CHECK: imull $12, %eax, %eax
imul $12, %eax

// CHECK: imull %ecx, %eax
imull %ecx, %eax


// rdar://8208481
// CHECK: outb	%al, $161
outb	%al, $161
// CHECK: outw	%ax, $128
outw	%ax, $128
// CHECK: inb	$161, %al
inb	$161, %al

// rdar://8017621
// CHECK: pushq	$1
push $1

// rdar://8017530
// CHECK: sldtw	4
sldt	4

// rdar://8208499
// CHECK: cmovnew	%bx, %ax
cmovnz %bx, %ax
// CHECK: cmovneq	%rbx, %rax
cmovnzq %rbx, %rax


// rdar://8407928
// CHECK: inb	$127, %al
// CHECK: inw	%dx, %ax
// CHECK: outb	%al, $127
// CHECK: outw	%ax, %dx
// CHECK: inl	%dx, %eax
inb	$0x7f
inw	%dx
outb	$0x7f
outw	%dx
inl	%dx


// PR8114
// CHECK: outb	%al, %dx
// CHECK: outw	%ax, %dx
// CHECK: outl	%eax, %dx

out %al, (%dx)
out %ax, (%dx)
outl %eax, (%dx)


// rdar://8431422

// CHECK: fxch	%st(1)
// CHECK: fucom	%st(1)
// CHECK: fucomp	%st(1)
// CHECK: faddp	%st(1)
// CHECK: faddp	%st(0)
// CHECK: fsubp	%st(1)
// CHECK: fsubrp	%st(1)
// CHECK: fmulp	%st(1)
// CHECK: fdivp	%st(1)
// CHECK: fdivrp	%st(1)

fxch
fucom
fucomp
faddp
faddp %st
fsubp
fsubrp
fmulp
fdivp
fdivrp

// CHECK: fcomi	%st(1), %st(0)
// CHECK: fcomi	%st(2), %st(0)
// CHECK: fucomi	%st(1), %st(0)
// CHECK: fucomi	%st(2), %st(0)
// CHECK: fucomi	%st(2), %st(0)

fcomi
fcomi	%st(2)
fucomi
fucomi	%st(2)
fucomi	%st(2), %st

// CHECK: fnstsw %ax
// CHECK: fnstsw %ax
// CHECK: fnstsw %ax
// CHECK: fnstsw %ax

fnstsw
fnstsw %ax
fnstsw %eax
fnstsw %al

// rdar://8431880
// CHECK: rclb	$1, %bl
// CHECK: rcll	$1, 3735928559(%ebx,%ecx,8)
// CHECK: rcrl	$1, %ecx
// CHECK: rcrl	$1, 305419896

rcl	%bl
rcll	0xdeadbeef(%ebx,%ecx,8)
rcr	%ecx
rcrl	0x12345678

// rdar://8418316
// CHECK: shldw	$1, %bx, %bx
// CHECK: shldw	$1, %bx, %bx
// CHECK: shrdw	$1, %bx, %bx
// CHECK: shrdw	$1, %bx, %bx

shld	%bx,%bx
shld	$1, %bx,%bx
shrd	%bx,%bx
shrd	$1, %bx,%bx

// CHECK: sldtl	%ecx
// CHECK: encoding: [0x0f,0x00,0xc1]
// CHECK: sldtw	%cx
// CHECK: encoding: [0x66,0x0f,0x00,0xc1]

sldt	%ecx
sldt	%cx

// CHECK: lcalll	*3135175374 
// CHECK: ljmpl	*3135175374
lcall	*0xbadeface
ljmp	*0xbadeface


// rdar://8444631
// CHECK: enter	$31438, $0
// CHECK: encoding: [0xc8,0xce,0x7a,0x00]
// CHECK: enter	$31438, $1
// CHECK: encoding: [0xc8,0xce,0x7a,0x01]
// CHECK: enter	$31438, $127
// CHECK: encoding: [0xc8,0xce,0x7a,0x7f]
enter $0x7ace,$0
enter $0x7ace,$1
enter $0x7ace,$0x7f


// rdar://8456389
// CHECK: fstps	(%eax)
// CHECK: encoding: [0xd9,0x18]
fstp	(%eax)

// rdar://8456364
// CHECK: movw	%cs, %ax
mov %CS, %ax

// rdar://8456391
fcmovb %st(1), %st(0)   // CHECK: fcmovb	%st(1), %st(0)
fcmove %st(1), %st(0)   // CHECK: fcmove	%st(1), %st(0)
fcmovbe %st(1), %st(0)  // CHECK: fcmovbe	%st(1), %st(0)
fcmovu %st(1), %st(0)   // CHECK: fcmovu	 %st(1), %st(0)

fcmovnb %st(1), %st(0)  // CHECK: fcmovnb	%st(1), %st(0)
fcmovne %st(1), %st(0)  // CHECK: fcmovne	%st(1), %st(0)
fcmovnbe %st(1), %st(0) // CHECK: fcmovnbe	%st(1), %st(0)
fcmovnu %st(1), %st(0)  // CHECK: fcmovnu	%st(1), %st(0)

fcmovnae %st(1), %st(0) // CHECK: fcmovb	%st(1), %st(0)
fcmovna %st(1), %st(0)  // CHECK: fcmovbe	%st(1), %st(0)

fcmovae %st(1), %st(0)  // CHECK: fcmovnb	%st(1), %st(0)
fcmova %st(1), %st(0)   // CHECK: fcmovnbe	%st(1), %st(0)

// rdar://8456417
.byte 88 + 1 & 15  // CHECK: .byte	9

// rdar://8456412
mov %rdx, %cr0
// CHECK: movq	%rdx, %cr0
// CHECK: encoding: [0x0f,0x22,0xc2]
mov %rdx, %cr4
// CHECK: movq	%rdx, %cr4
// CHECK: encoding: [0x0f,0x22,0xe2]
mov %rdx, %cr8
// CHECK: movq	%rdx, %cr8
// CHECK: encoding: [0x44,0x0f,0x22,0xc2]
mov %rdx, %cr15
// CHECK: movq	%rdx, %cr15
// CHECK: encoding: [0x44,0x0f,0x22,0xfa]

// rdar://8456371 - Handle commutable instructions written backward.
// CHECK: 	faddp	%st(1)
// CHECK:	fmulp	%st(2)
faddp %st, %st(1)
fmulp %st, %st(2)

// rdar://8468087 - Encode these accurately, they are not synonyms.
// CHECK: fmul	%st(0), %st(1)
// CHECK: encoding: [0xdc,0xc9]
// CHECK: fmul	%st(1)
// CHECK: encoding: [0xd8,0xc9]
fmul %st, %st(1)
fmul %st(1), %st

// CHECK: fadd	%st(0), %st(1)
// CHECK: encoding: [0xdc,0xc1]
// CHECK: fadd	%st(1)
// CHECK: encoding: [0xd8,0xc1]
fadd %st, %st(1)
fadd %st(1), %st


// rdar://8416805
// CHECK: xorb	%al, %al
// CHECK: encoding: [0x30,0xc0]
// CHECK: xorw	%di, %di
// CHECK: encoding: [0x66,0x31,0xff]
// CHECK: xorl	%esi, %esi
// CHECK: encoding: [0x31,0xf6]
// CHECK: xorq	%rsi, %rsi
// CHECK: encoding: [0x48,0x31,0xf6]
clrb    %al
clr    %di
clr    %esi
clr    %rsi


