// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

movl	foo(%rip), %eax
// CHECK: movl	foo(%rip), %eax
// CHECK: encoding: [0x8b,0x05,A,A,A,A]
// CHECK: fixup A - offset: 2, value: foo-4, kind: reloc_riprel_4byte

movb	$12, foo(%rip)
// CHECK: movb	$12, foo(%rip)
// CHECK: encoding: [0xc6,0x05,A,A,A,A,0x0c]
// CHECK:    fixup A - offset: 2, value: foo-5, kind: reloc_riprel_4byte

movw	$12, foo(%rip)
// CHECK: movw	$12, foo(%rip)
// CHECK: encoding: [0x66,0xc7,0x05,A,A,A,A,0x0c,0x00]
// CHECK:    fixup A - offset: 3, value: foo-6, kind: reloc_riprel_4byte

movl	$12, foo(%rip)
// CHECK: movl	$12, foo(%rip)
// CHECK: encoding: [0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 2, value: foo-8, kind: reloc_riprel_4byte

movq	$12, foo(%rip)
// CHECK:  movq	$12, foo(%rip)
// CHECK: encoding: [0x48,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 3, value: foo-8, kind: reloc_riprel_4byte

// CHECK: addq	$-424, %rax
// CHECK: encoding: [0x48,0x05,0x58,0xfe,0xff,0xff]
addq $-424, %rax


// CHECK: movq	_foo@GOTPCREL(%rip), %rax
// CHECK:  encoding: [0x48,0x8b,0x05,A,A,A,A]
// CHECK:  fixup A - offset: 3, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%rip), %rax

// CHECK: movq	_foo@GOTPCREL(%rip), %r14
// CHECK:  encoding: [0x4c,0x8b,0x35,A,A,A,A]
// CHECK:  fixup A - offset: 3, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%rip), %r14


// CHECK: movq	(%r13,%rax,8), %r13
// CHECK:  encoding: [0x4d,0x8b,0x6c,0xc5,0x00]
movq 0x00(%r13,%rax,8),%r13

// CHECK: testq	%rax, %rbx
// CHECK:  encoding: [0x48,0x85,0xd8]
testq %rax, %rbx

// CHECK: cmpq	%rbx, %r14
// CHECK:   encoding: [0x49,0x39,0xde]
        cmpq %rbx, %r14

// rdar://7947167

movsq
// CHECK: movsq
// CHECK:   encoding: [0x48,0xa5]

movsl
// CHECK: movsl
// CHECK:   encoding: [0xa5]

stosq
// CHECK: stosq
// CHECK:   encoding: [0x48,0xab]
stosl
// CHECK: stosl
// CHECK:   encoding: [0xab]


// Not moffset forms of moves, they are x86-32 only! rdar://7947184
movb	0, %al    // CHECK: movb 0, %al # encoding: [0x8a,0x04,0x25,0x00,0x00,0x00,0x00]
movw	0, %ax    // CHECK: movw 0, %ax # encoding: [0x66,0x8b,0x04,0x25,0x00,0x00,0x00,0x00]
movl	0, %eax   // CHECK: movl 0, %eax # encoding: [0x8b,0x04,0x25,0x00,0x00,0x00,0x00]

// CHECK: pushfq	# encoding: [0x9c]
        pushf
// CHECK: pushfq	# encoding: [0x9c]
        pushfq
// CHECK: popfq	        # encoding: [0x9d]
        popf
// CHECK: popfq	        # encoding: [0x9d]
        popfq

// CHECK: movabsq $-281474976710654, %rax
// CHECK: encoding: [0x48,0xb8,0x02,0x00,0x00,0x00,0x00,0x00,0xff,0xff]
        movabsq $0xFFFF000000000002, %rax

// CHECK: movq $-281474976710654, %rax
// CHECK: encoding: [0x48,0xb8,0x02,0x00,0x00,0x00,0x00,0x00,0xff,0xff]
        movq $0xFFFF000000000002, %rax

// CHECK: movq $-65536, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x00,0x00,0xff,0xff]
        movq $0xFFFFFFFFFFFF0000, %rax

// CHECK: movq $-256, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x00,0xff,0xff,0xff]
        movq $0xFFFFFFFFFFFFFF00, %rax

// CHECK: movq $10, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x0a,0x00,0x00,0x00]
        movq $10, %rax

// rdar://8014869
//
// CHECK: ret
// CHECK:  encoding: [0xc3]
        retq

// CHECK: sete %al
// CHECK: encoding: [0x0f,0x94,0xc0]
        setz %al

// CHECK: setne %al
// CHECK: encoding: [0x0f,0x95,0xc0]
        setnz %al

// CHECK: je 0
// CHECK: encoding: [0x74,A]
        jz 0

// CHECK: jne
// CHECK: encoding: [0x75,A]
        jnz 0

// rdar://8017515
btq $0x01,%rdx
// CHECK: btq	$1, %rdx
// CHECK:  encoding: [0x48,0x0f,0xba,0xe2,0x01]

//rdar://8017633
// CHECK: movzbl	%al, %esi
// CHECK:  encoding: [0x0f,0xb6,0xf0]
        movzx %al, %esi

// CHECK: movzbq	%al, %rsi
// CHECK:  encoding: [0x48,0x0f,0xb6,0xf0]
        movzx %al, %rsi

// CHECK: movzbq	(%rsp), %rsi
// CHECK:  encoding: [0x48,0x0f,0xb6,0x34,0x24]
        movzx 0(%rsp), %rsi


// rdar://7873482
// CHECK: [0x65,0x8b,0x04,0x25,0x7c,0x00,0x00,0x00]
        movl	%gs:124, %eax

// CHECK: jmpq *8(%rax)
// CHECK:   encoding: [0xff,0x60,0x08]
	jmp	*8(%rax)

// CHECK: btq $61, -216(%rbp)
// CHECK:   encoding: [0x48,0x0f,0xba,0xa5,0x28,0xff,0xff,0xff,0x3d]
	btq	$61, -216(%rbp)
