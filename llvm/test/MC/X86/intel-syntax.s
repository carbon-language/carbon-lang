// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

_test:
	xor	EAX, EAX
	ret

_main:
// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257
// CHECK:	movl	$258, 4(%rsp)
	mov	DWORD PTR [RSP + 4], 258
// CHECK:	movq	$123, -16(%rsp)
	mov	QWORD PTR [RSP - 16], 123
// CHECK:	movb	$97, -17(%rsp)
	mov	BYTE PTR [RSP - 17], 97
// CHECK:	movl	-4(%rsp), %eax
	mov	EAX, DWORD PTR [RSP - 4]
// CHECK:	movq    (%rsp), %rax
	mov     RAX, QWORD PTR [RSP]
// CHECK:	movl	$-4, -4(%rsp)
	mov	DWORD PTR [RSP - 4], -4
// CHECK:	movq	0, %rcx
	mov	RCX, QWORD PTR [0]
// CHECK:	movl	-24(%rsp,%rax,4), %eax	
	mov	EAX, DWORD PTR [RSP + 4*RAX - 24]
// CHECK:	movb	%dil, (%rdx,%rcx)
	mov	BYTE PTR [RDX + RCX], DIL
// CHECK:	movzwl	2(%rcx), %edi
	movzx	EDI, WORD PTR [RCX + 2]
// CHECK:	callq	_test
	call	_test
// CHECK:	andw	$12,	%ax
	and	ax, 12
// CHECK:	andw	$-12,	%ax
	and	ax, -12
// CHECK:	andw	$257,	%ax
	and	ax, 257
// CHECK:	andw	$-257,	%ax
	and	ax, -257
// CHECK:	andl	$12,	%eax
	and	eax, 12
// CHECK:	andl	$-12,	%eax
	and	eax, -12
// CHECK:	andl	$257,	%eax
	and	eax, 257
// CHECK:	andl	$-257,	%eax
	and	eax, -257
// CHECK:	andq	$12,	%rax
	and	rax, 12
// CHECK:	andq	$-12,	%rax
	and	rax, -12
// CHECK:	andq	$257,	%rax
	and	rax, 257
// CHECK:	andq	$-257,	%rax
	and	rax, -257
// CHECK:	fld	%st(0)
	fld	ST(0)
// CHECK:	movl	%fs:(%rdi), %eax
    mov EAX, DWORD PTR FS:[RDI]
// CHECK: leal (,%rdi,4), %r8d
    lea R8D, DWORD PTR [4*RDI]
// CHECK: movl _fnan(,%ecx,4), %ecx
    mov ECX, DWORD PTR [4*ECX + _fnan]
// CHECK: movq %fs:320, %rax
    mov RAX, QWORD PTR FS:[320]
// CHECK: vpgatherdd %xmm8, (%r15,%xmm9,2), %xmm1
    vpgatherdd XMM10, DWORD PTR [R15 + 2*XMM9], XMM8
// CHECK: movsd	-8, %xmm5
    movsd   XMM5, QWORD PTR [-8]
// CHECK: movl %ecx, (%eax)
    mov [eax], ecx
// CHECK: movl %ecx, (,%ebx,4)
    mov [4*ebx], ecx
 // CHECK:   movl %ecx, (,%ebx,4)
    mov [ebx*4], ecx
// CHECK: movl %ecx, 1024
    mov [1024], ecx
// CHECK: movl %ecx, 4132
    mov [0x1024], ecx
// CHECK: movl %ecx, 32        
    mov [16 + 16], ecx
// CHECK: movl %ecx, 0
    mov [16 - 16], ecx        
// CHECK: movl %ecx, 32        
    mov [16][16], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax + 4*ebx], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax + ebx*4], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [4*ebx + eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [ebx*4 + eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax][4*ebx], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax][ebx*4], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [4*ebx][eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [ebx*4][eax], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [eax + 12], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [12 + eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax + 16 + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + eax + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + 16 + eax], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [eax][12], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [12][eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16 + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax + 16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][eax + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + eax][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][16 + eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + 16][eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][eax][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][16][eax], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [4*ebx + 16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [4*ebx][16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [ebx*4][16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16 + ebx*4], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16][4*ebx], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 4*ebx + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + 16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][4*ebx + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 4*ebx][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + 16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][4*ebx][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16 + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + 16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16 + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + ebx*4][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + 16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][ebx*4][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 16], ecx

// CHECK: prefetchnta 12800(%esi)
    prefetchnta [esi + (200*64)]
// CHECK: prefetchnta 32(%esi)
    prefetchnta [esi + (64/2)]
// CHECK: prefetchnta 128(%esi)
    prefetchnta [esi + (64/2*4)]
// CHECK: prefetchnta 8(%esi)
    prefetchnta [esi + (64/(2*4))]
// CHECK: prefetchnta 48(%esi)
    prefetchnta [esi + (64/(2*4)+40)]

// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][4*ebx - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + 4*ebx - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [12 + eax + (4*ebx) - 2*14], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2*2*2*2], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - (2*8)], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2 * 8 + 4 - 4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ebx*4 - 2 * 8 + 4 - 4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ebx*4 - 2 * ((8 + 4) - 4)], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [-2 * ((8 + 4) - 4) + eax + ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [((-2) * ((8 + 4) - 4)) + eax + ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ((-2) * ((8 + 4) - 4)) + ebx*4], ecx

    ret
