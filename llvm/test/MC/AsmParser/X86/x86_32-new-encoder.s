// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

	pause
// CHECK: pause
// CHECK: encoding: [0xf3,0x90]
	sfence
// CHECK: sfence
// CHECK: encoding: [0x0f,0xae,0xf8]
	lfence
// CHECK: lfence
// CHECK: encoding: [0x0f,0xae,0xe8]
	mfence
// CHECK: mfence
// CHECK: encoding: [0x0f,0xae,0xf0]
	monitor
// CHECK: monitor
// CHECK: encoding: [0x0f,0x01,0xc8]
	mwait
// CHECK: mwait
// CHECK: encoding: [0x0f,0x01,0xc9]

	vmcall
// CHECK: vmcall
// CHECK: encoding: [0x0f,0x01,0xc1]
	vmlaunch
// CHECK: vmlaunch
// CHECK: encoding: [0x0f,0x01,0xc2]
	vmresume
// CHECK: vmresume
// CHECK: encoding: [0x0f,0x01,0xc3]
	vmxoff
// CHECK: vmxoff
// CHECK: encoding: [0x0f,0x01,0xc4]
	swapgs
// CHECK: swapgs
// CHECK: encoding: [0x0f,0x01,0xf8]

rdtscp
// CHECK: rdtscp
// CHECK:  encoding: [0x0f,0x01,0xf9]


// CHECK: movl	%eax, 16(%ebp)          # encoding: [0x89,0x45,0x10]
	movl	%eax, 16(%ebp)
// CHECK: movl	%eax, -16(%ebp)          # encoding: [0x89,0x45,0xf0]
	movl	%eax, -16(%ebp)

// CHECK: testb	%bl, %cl                # encoding: [0x84,0xcb]
        testb %bl, %cl

// CHECK: cmpl	%eax, %ebx              # encoding: [0x39,0xc3]
        cmpl %eax, %ebx

// CHECK: addw	%ax, %ax                # encoding: [0x66,0x01,0xc0]
        addw %ax, %ax

// CHECK: shrl	%eax                    # encoding: [0xd1,0xe8]
        shrl $1, %eax

// moffset forms of moves, rdar://7947184
movb	0, %al    // CHECK: movb 0, %al  # encoding: [0xa0,A,A,A,A]
movw	0, %ax    // CHECK: movw 0, %ax  # encoding: [0x66,0xa1,A,A,A,A]
movl	0, %eax   // CHECK: movl 0, %eax  # encoding: [0xa1,A,A,A,A]

// rdar://7973775
into
// CHECK: into
// CHECK:  encoding: [0xce]
int3
// CHECK: int3
// CHECK:  encoding: [0xcc]
int $4
// CHECK: int $4
// CHECK:  encoding: [0xcd,0x04]
int $255
// CHECK: int $255
// CHECK:  encoding: [0xcd,0xff]

// CHECK: pushfl	# encoding: [0x9c]
        pushf
// CHECK: pushfl	# encoding: [0x9c]
        pushfl
// CHECK: popfl	        # encoding: [0x9d]
        popf
// CHECK: popfl	        # encoding: [0x9d]
        popfl

// rdar://8014869
retl
// CHECK: ret
// CHECK:  encoding: [0xc3]

// rdar://7973854
// CHECK: cmoval	%eax, %edx
// CHECK:  encoding: [0x0f,0x47,0xd0]
        	cmoval	%eax,%edx

// CHECK: cmovael	%eax, %edx
// CHECK:  encoding: [0x0f,0x43,0xd0]
        	cmovael	%eax,%edx

// CHECK: cmovbel	%eax, %edx
// CHECK:  encoding: [0x0f,0x46,0xd0]
        	cmovbel	%eax,%edx

// CHECK: cmovbl	%eax, %edx
// CHECK:  encoding: [0x0f,0x42,0xd0]
        	cmovbl	%eax,%edx

// CHECK: cmovbel	%eax, %edx
// CHECK:  encoding: [0x0f,0x46,0xd0]
        	cmovbel	%eax,%edx

// CHECK: cmovbl	%eax, %edx
// CHECK:  encoding: [0x0f,0x42,0xd0]
        	cmovcl	%eax,%edx

// CHECK: cmovel	%eax, %edx
// CHECK:  encoding: [0x0f,0x44,0xd0]
        	cmovel	%eax,%edx

// CHECK: cmovgl	%eax, %edx
// CHECK:  encoding: [0x0f,0x4f,0xd0]
        	cmovgl	%eax,%edx

// CHECK: cmovgel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4d,0xd0]
        	cmovgel	%eax,%edx

// CHECK: cmovll	%eax, %edx
// CHECK:  encoding: [0x0f,0x4c,0xd0]
        	cmovll	%eax,%edx

// CHECK: cmovlel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4e,0xd0]
        	cmovlel	%eax,%edx

// CHECK: cmovbel	%eax, %edx
// CHECK:  encoding: [0x0f,0x46,0xd0]
        	cmovnal	%eax,%edx

// CHECK: cmovnel	%eax, %edx
// CHECK:  encoding: [0x0f,0x45,0xd0]
        	cmovnel	%eax,%edx

// CHECK: cmovael	%eax, %edx
// CHECK:  encoding: [0x0f,0x43,0xd0]
        	cmovnbl	%eax,%edx

// CHECK: cmoval	%eax, %edx
// CHECK:  encoding: [0x0f,0x47,0xd0]
        	cmovnbel	%eax,%edx

// CHECK: cmovael	%eax, %edx
// CHECK:  encoding: [0x0f,0x43,0xd0]
        	cmovncl	%eax,%edx

// CHECK: cmovnel	%eax, %edx
// CHECK:  encoding: [0x0f,0x45,0xd0]
        	cmovnel	%eax,%edx

// CHECK: cmovlel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4e,0xd0]
        	cmovngl	%eax,%edx

// CHECK: cmovgel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4d,0xd0]
        	cmovnl	%eax,%edx

// CHECK: cmovnel	%eax, %edx
// CHECK:  encoding: [0x0f,0x45,0xd0]
        	cmovnel	%eax,%edx

// CHECK: cmovlel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4e,0xd0]
        	cmovngl	%eax,%edx

// CHECK: cmovll	%eax, %edx
// CHECK:  encoding: [0x0f,0x4c,0xd0]
        	cmovngel	%eax,%edx

// CHECK: cmovgel	%eax, %edx
// CHECK:  encoding: [0x0f,0x4d,0xd0]
        	cmovnll	%eax,%edx

// CHECK: cmovgl	%eax, %edx
// CHECK:  encoding: [0x0f,0x4f,0xd0]
        	cmovnlel	%eax,%edx

// CHECK: cmovnol	%eax, %edx
// CHECK:  encoding: [0x0f,0x41,0xd0]
        	cmovnol	%eax,%edx

// CHECK: cmovnpl	%eax, %edx
// CHECK:  encoding: [0x0f,0x4b,0xd0]
        	cmovnpl	%eax,%edx

// CHECK: cmovnsl	%eax, %edx
// CHECK:  encoding: [0x0f,0x49,0xd0]
        	cmovnsl	%eax,%edx

// CHECK: cmovnel	%eax, %edx
// CHECK:  encoding: [0x0f,0x45,0xd0]
        	cmovnzl	%eax,%edx

// CHECK: cmovol	%eax, %edx
// CHECK:  encoding: [0x0f,0x40,0xd0]
        	cmovol	%eax,%edx

// CHECK: cmovpl	%eax, %edx
// CHECK:  encoding: [0x0f,0x4a,0xd0]
        	cmovpl	%eax,%edx

// CHECK: cmovsl	%eax, %edx
// CHECK:  encoding: [0x0f,0x48,0xd0]
        	cmovsl	%eax,%edx

// CHECK: cmovel	%eax, %edx
// CHECK:  encoding: [0x0f,0x44,0xd0]
        	cmovzl	%eax,%edx

// CHECK: cmpps	$0, %xmm0, %xmm1
// CHECK: encoding: [0x0f,0xc2,0xc8,0x00]
        cmpps $0, %xmm0, %xmm1
// CHECK:	cmpps	$0, (%eax), %xmm1
// CHECK: encoding: [0x0f,0xc2,0x08,0x00]
        cmpps $0, 0(%eax), %xmm1
// CHECK:	cmppd	$0, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0xc2,0xc8,0x00]
        cmppd $0, %xmm0, %xmm1
// CHECK:	cmppd	$0, (%eax), %xmm1
// CHECK: encoding: [0x66,0x0f,0xc2,0x08,0x00]
        cmppd $0, 0(%eax), %xmm1
// CHECK:	cmpss	$0, %xmm0, %xmm1
// CHECK: encoding: [0xf3,0x0f,0xc2,0xc8,0x00]
        cmpss $0, %xmm0, %xmm1
// CHECK:	cmpss	$0, (%eax), %xmm1
// CHECK: encoding: [0xf3,0x0f,0xc2,0x08,0x00]
        cmpss $0, 0(%eax), %xmm1
// CHECK:	cmpsd	$0, %xmm0, %xmm1
// CHECK: encoding: [0xf2,0x0f,0xc2,0xc8,0x00]
        cmpsd $0, %xmm0, %xmm1
// CHECK:	cmpsd	$0, (%eax), %xmm1
// CHECK: encoding: [0xf2,0x0f,0xc2,0x08,0x00]
        cmpsd $0, 0(%eax), %xmm1

// Check matching of instructions which embed the SSE comparison code.

// CHECK: cmpps $0, %xmm0, %xmm1
// CHECK: encoding: [0x0f,0xc2,0xc8,0x00]
        cmpeqps %xmm0, %xmm1

// CHECK: cmppd $1, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0xc2,0xc8,0x01]
        cmpltpd %xmm0, %xmm1

// CHECK: cmpss $2, %xmm0, %xmm1
// CHECK: encoding: [0xf3,0x0f,0xc2,0xc8,0x02]
        cmpless %xmm0, %xmm1

// CHECK: cmppd $3, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0xc2,0xc8,0x03]
        cmpunordpd %xmm0, %xmm1

// CHECK: cmpps $4, %xmm0, %xmm1
// CHECK: encoding: [0x0f,0xc2,0xc8,0x04]
        cmpneqps %xmm0, %xmm1

// CHECK: cmppd $5, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0xc2,0xc8,0x05]
        cmpnltpd %xmm0, %xmm1

// CHECK: cmpss $6, %xmm0, %xmm1
// CHECK: encoding: [0xf3,0x0f,0xc2,0xc8,0x06]
        cmpnless %xmm0, %xmm1

// CHECK: cmpsd $7, %xmm0, %xmm1
// CHECK: encoding: [0xf2,0x0f,0xc2,0xc8,0x07]
        cmpordsd %xmm0, %xmm1

// rdar://7995856
// CHECK: fmul	%st(0)
// CHECK:  encoding: [0xd8,0xc8]
        fmul %st(0), %st

// CHECK: fadd	%st(0)
// CHECK:  encoding: [0xd8,0xc0]
        fadd %st(0), %st

// CHECK: fsub	%st(0)
// CHECK:  encoding: [0xd8,0xe0]
        fsub %st(0), %st

// CHECK: fsubr	%st(0)
// CHECK:  encoding: [0xd8,0xe8]
        fsubr %st(0), %st

// CHECK: fdivr	%st(0)
// CHECK:  encoding: [0xd8,0xf8]
        fdivr %st(0), %st

// CHECK: fdiv	%st(0)
// CHECK:  encoding: [0xd8,0xf0]
        fdiv %st(0), %st

// radr://8017519
// CHECK: movl	%cs, %eax
// CHECK:  encoding: [0x8c,0xc8]
        movl %cs, %eax

// CHECK: movw	%cs, %ax
// CHECK:  encoding: [0x66,0x8c,0xc8]
        movw %cs, %ax

// CHECK: movl	%cs, (%eax)
// CHECK:  encoding: [0x8c,0x08]
        movl %cs, (%eax)

// CHECK: movw	%cs, (%eax)
// CHECK:  encoding: [0x66,0x8c,0x08]
        movw %cs, (%eax)

// CHECK: movl	%eax, %cs
// CHECK:  encoding: [0x8e,0xc8]
        movl %eax, %cs

// CHECK: movl	(%eax), %cs
// CHECK:  encoding: [0x8e,0x08]
        movl (%eax), %cs

// CHECK: movw	(%eax), %cs
// CHECK:  encoding: [0x66,0x8e,0x08]
        movw (%eax), %cs

// radr://8033374
// CHECK: movl	%cr0, %eax
// CHECK:  encoding: [0x0f,0x20,0xc0]
        movl %cr0,%eax

// CHECK: movl	%cr1, %eax
// CHECK:  encoding: [0x0f,0x20,0xc8]
        movl %cr1,%eax

// CHECK: movl	%cr2, %eax
// CHECK:  encoding: [0x0f,0x20,0xd0]
        movl %cr2,%eax

// CHECK: movl	%cr3, %eax
// CHECK:  encoding: [0x0f,0x20,0xd8]
        movl %cr3,%eax

// CHECK: movl	%cr4, %eax
// CHECK:  encoding: [0x0f,0x20,0xe0]
        movl %cr4,%eax

// CHECK: movl	%dr0, %eax
// CHECK:  encoding: [0x0f,0x21,0xc0]
        movl %dr0,%eax

// CHECK: movl	%dr1, %eax
// CHECK:  encoding: [0x0f,0x21,0xc8]
        movl %dr1,%eax

// CHECK: movl	%dr1, %eax
// CHECK:  encoding: [0x0f,0x21,0xc8]
        movl %dr1,%eax

// CHECK: movl	%dr2, %eax
// CHECK:  encoding: [0x0f,0x21,0xd0]
        movl %dr2,%eax

// CHECK: movl	%dr3, %eax
// CHECK:  encoding: [0x0f,0x21,0xd8]
        movl %dr3,%eax

// CHECK: movl	%dr4, %eax
// CHECK:  encoding: [0x0f,0x21,0xe0]
        movl %dr4,%eax

// CHECK: movl	%dr5, %eax
// CHECK:  encoding: [0x0f,0x21,0xe8]
        movl %dr5,%eax

// CHECK: movl	%dr6, %eax
// CHECK:  encoding: [0x0f,0x21,0xf0]
        movl %dr6,%eax

// CHECK: movl	%dr7, %eax
// CHECK:  encoding: [0x0f,0x21,0xf8]
        movl %dr7,%eax

// radr://8017522
// CHECK: wait
// CHECK:  encoding: [0x9b]
	fwait

// rdar://7873482
// CHECK: [0x65,0x8b,0x05,0x7c,0x00,0x00,0x00]
// FIXME: This is a correct bug poor encoding: Use 65 a1 7c 00 00 00 
        movl	%gs:124, %eax

// CHECK: pusha
// CHECK:  encoding: [0x60]
        	pusha

// CHECK: popa
// CHECK:  encoding: [0x61]
        	popa

// CHECK: pushal
// CHECK:  encoding: [0x60]
        	pushal

// CHECK: popal
// CHECK:  encoding: [0x61]
        	popal

// CHECK: jmpl *8(%eax)
// CHECK:   encoding: [0xff,0x60,0x08]
	jmp	*8(%eax)

// PR7465
// CHECK: lcalll $2, $4660
// CHECK:   encoding: [0x9a,0x34,0x12,0x00,0x00,0x02,0x00]
lcalll $0x2, $0x1234
