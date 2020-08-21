# This reproduces a bug with shrink wrapping when trying to split critical
# edges originating at the same basic block.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: strip --strip-unneeded %t.o
# RUN: %host_cc %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:     -frame-opt=all -debug-only=shrinkwrapping \
# RUN:     -simplify-conditional-tail-calls=false -eliminate-unreachable=false \
# RUN:    2>&1 | FileCheck %s

# CHECK: - Now handling FrontierBB .LFT{{.*}}
# CHECK-NEXT: - Dest : .Ltmp{{.*}}
# CHECK-NEXT: - Update frontier with .LSplitEdge{{.*}}
# CHECK-NEXT: - Dest : .Ltmp{{.*}}
# CHECK-NEXT: - Append frontier .LSplitEdge{{.*}}

  .text
  .globl main
  .type main, %function
  .p2align  4
  .globl _start
_start:
main:
# FDATA: 0 [unknown] 0 1 main 0 0 186
LBB074208:
  	pushq	%rbp
  	movl	%esi, %eax
  	shrw	$0x8, %ax
  	movq	%rsp, %rbp
  	pushq	%r14
  	pushq	%r13
  	pushq	%r12
  	pushq	%rbx
  	movl	%edi, %ebx
  	cmpb	$0xd, %al
LBB074208j:
  	ja	Ltmp12774
# FDATA: 1 main #LBB074208j# 1 main #Ltmp12774# 0 0
# FDATA: 1 main #LBB074208j# 1 main #LFT773198# 0 137

LFT773198:
  	movzbl	%al, %ecx
LFT773198j:
  	jmpq	*"JUMP_TABLE0"(,%rcx,8)
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12774# 5 154
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12780# 2 3
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12779# 2 2
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12775# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12776# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12778# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12783# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12777# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12781# 0 0
# FDATA: 1 main #LFT773198j# 1 main #Ltmp12782# 0 0

Ltmp12774:
  	cmpw	$0xff, %si
Ltmp12774j:
  	jne	Ltmp1027620
# FDATA: 1 main #Ltmp12774j# 1 main #Ltmp1027620# 65 113
# FDATA: 1 main #Ltmp12774j# 1 main #LFT773204# 0 44

LFT773204:
  	cmpb	$0xc, %bl
LFT773204j:
  	je	Ltmp1027621
# FDATA: 1 main #LFT773204j# 1 main #Ltmp1027621# 0 0
# FDATA: 1 main #LFT773204j# 1 main #Ltmp1027620# 0 44

Ltmp1027620:
  	cmpb	$0xe, %bl
Ltmp1027620j:
  	je	Ltmp1027622
# FDATA: 1 main #Ltmp1027620j# 1 main #Ltmp1027622# 0 0
# FDATA: 1 main #Ltmp1027620j# 1 main #LFT773251# 0 155

LFT773251:
  	cmpw	$0xfd, %si
LFT773251j:
  	je	Ltmp1027623
# FDATA: 1 main #LFT773251j# 1 main #Ltmp1027623# 0 0
# FDATA: 1 main #LFT773251j# 1 main #LFT773287# 0 155

LFT773287:
  	cmpw	$0x3, %si
LFT773287j:
  	je	Ltmp1027624
# FDATA: 1 main #LFT773287j# 1 main #Ltmp1027624# 58 66
# FDATA: 1 main #LFT773287j# 1 main #LFT773295# 0 91

LFT773295:
  	cmpb	%sil, %bl
LFT773295j:
  	jne	Ltmp1027625
# FDATA: 1 main #LFT773295j# 1 main #Ltmp1027625# 17 21
# FDATA: 1 main #LFT773295j# 1 main #Ltmp12775# 0 71

Ltmp12775:
  	xorl	%eax, %eax
# FDATA: 1 main #Ltmp12775# 1 main #Ltmp1027626# 0 74

Ltmp1027626:
  	popq	%rbx
  	popq	%r12
  	popq	%r13
  	popq	%r14
  	popq	%rbp
  	retq

Ltmp1027625:
  	testb	%sil, %bl
Ltmp1027625j:
  	jns	Ltmp12784
# FDATA: 1 main #Ltmp1027625j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027625j# 1 main #LFT773360# 0 21

LFT773360:
  	xorl	%ebx, %esi
  	xorl	%eax, %eax
  	andb	$-0x2, %sil
  	setne	%al
LFT773360j:
  	jmp	Ltmp1027626
# FDATA: 1 main #LFT773360j# 1 main #Ltmp1027626# 0 22

Ltmp12776:
  	movl	$0x2, %eax
  	cmpb	$0x3, %dil
Ltmp12776j:
  	je	Ltmp1027626
# FDATA: 1 main #Ltmp12776j# 1 main #Ltmp1027626# 0 0
# FDATA: 1 main #Ltmp12776j# 1 main #Ltmp12784# 0 0

Ltmp12784:
  	movl	$0x1, %eax
Ltmp12784j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp12784j# 1 main #Ltmp1027626# 0 0

Ltmp12779:
  	xorl	%edx, %edx
  	cmpb	$0x6, %dil
Ltmp12779j:
  	je	Ltmp1027627
# FDATA: 1 main #Ltmp12779j# 1 main #Ltmp1027627# 0 1
# FDATA: 1 main #Ltmp12779j# 1 main #LFT773364# 0 1

LFT773364:
  	cmpb	$-0x3, %dil
LFT773364j:
  	setbe	%dl
# FDATA: 1 main #LFT773364j# 1 main #Ltmp1027627# 0 1

Ltmp1027627:
  	movzbl	%dl, %eax
Ltmp1027627j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp1027627j# 1 main #Ltmp1027626# 0 2

Ltmp12780:
  	cmpb	$0xe, %dil
Ltmp12780j:
  	je	Ltmp1027628
# FDATA: 1 main #Ltmp12780j# 1 main #Ltmp1027628# 0 0
# FDATA: 1 main #Ltmp12780j# 1 main #LFT773437# 0 3

LFT773437:
  	cmpb	$0x0, data1
LFT773437j:
  	jne	Ltmp1027629
# FDATA: 1 main #LFT773437j# 1 main #Ltmp1027629# 0 0
# FDATA: 1 main #LFT773437j# 1 main #LFT773496# 0 3

LFT773496:
  	orl	$0x1, %ebx
  	leal	0xd(%rbx), %eax
  	testb	$-0x3, %al
LFT773496j:
  	je	Ltmp12775
LFT773496j2:
  	jmp	Ltmp12784
# FDATA: 1 main #LFT773496j# 1 main #Ltmp12775# 0 3
# FDATA: 1 main #LFT773496j2# 1 main #Ltmp12784# 0 0

Ltmp12778:
  	leal	-0x6(%rdi), %esi
  	xorl	%eax, %eax
  	andb	$-0x3, %sil
  	setne	%al
Ltmp12778j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp12778j# 1 main #Ltmp1027626# 0 0

Ltmp12783:
  	xorl	%eax, %eax
  	cmpb	$0x2, %dil
  	sete	%al
Ltmp12783j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp12783j# 1 main #Ltmp1027626# 0 0

Ltmp1027624:
  	cmpb	$0x3, %bl
Ltmp1027624j:
  	je	Ltmp12786
# FDATA: 1 main #Ltmp1027624j# 1 main #Ltmp12786# 23 47
# FDATA: 1 main #Ltmp1027624j# 1 main #Ltmp1027654# 0 27

Ltmp1027654:
  	movl	0x8(%rdx), %r12d
  	leaq	0x10(%rdx), %r14
  	movq	%r14, %rdi
  	movq	%r12, %r13
  	addq	%r14, %r12
  	movq	%r12, %rsi
  	callq	"func1"
  	testb	%al, %al
Ltmp1027654j:
  	je	Ltmp1027630
# FDATA: 1 main #Ltmp1027654j# 1 main #Ltmp1027630# 0 0
# FDATA: 1 main #Ltmp1027654j# 1 main #Ltmp1027632# 0 8

Ltmp1027632:
  	leal	0xe(%rbx), %edx
  	cmpb	$0x1c, %dl
Ltmp1027632j:
  	ja	Ltmp12786
# FDATA: 1 main #Ltmp1027632j# 1 main #Ltmp12786# 0 0
# FDATA: 1 main #Ltmp1027632j# 1 main #LFT773556# 0 8

LFT773556:
  	movzbl	%dl, %ebx
LFT773556j:
  	jmpq	*"JUMP_TABLE1"(,%rbx,8)
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12785# 0 8
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12786# 0 0
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12787# 0 0
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12788# 0 0
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12790# 0 0
# FDATA: 1 main #LFT773556j# 1 main #Ltmp12789# 0 0

Ltmp12785:
  	movq	%r14, %rdi
  	movq	%r12, %rsi
  	callq	"func1"
  	xorl	$0x1, %eax
  	movzbl	%al, %eax
Ltmp12785j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp12785j# 1 main #Ltmp1027626# 0 16

Ltmp1027630:
  	movq	data3, %rsi
  	testq	%rsi, %rsi
Ltmp1027630j:
  	je	Ltmp1027631
# FDATA: 1 main #Ltmp1027630j# 1 main #Ltmp1027631# 0 0
# FDATA: 1 main #Ltmp1027630j# 1 main #LFT773658# 0 0

LFT773658:
  	movl	0x8(%rsi), %r9d
  	leaq	0x10(%rsi), %r10
  	movq	%r10, %rsi
LFT773658j:
  	addq	%r10, %r9
# FDATA: 1 main #LFT773658j# 1 main #Ltmp1027644# 0 0

Ltmp1027644:
  	subq	%rsi, %r9
  	movl	%r13d, %r11d
  	cmpq	%r11, %r9
Ltmp1027644j:
  	jne	Ltmp12786
# FDATA: 1 main #Ltmp1027644j# 1 main #Ltmp12786# 0 0
# FDATA: 1 main #Ltmp1027644j# 1 main #LFT773703# 0 0

LFT773703:
  	movq	%r14, %rdi
  	callq	func2
  	testl	%eax, %eax
LFT773703j:
  	je	Ltmp1027632
# FDATA: 1 main #LFT773703j# 1 main #Ltmp1027632# 0 0
# FDATA: 1 main #LFT773703j# 1 main #Ltmp12786# 0 0

Ltmp12786:
  	movl	$0x2, %eax
Ltmp12786j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp12786j# 1 main #Ltmp1027626# 0 49

Ltmp12787:
  	movq	data4, %rsi
  	testq	%rsi, %rsi
Ltmp12787j:
  	je	Ltmp1027633
# FDATA: 1 main #Ltmp12787j# 1 main #Ltmp1027633# 0 0
# FDATA: 1 main #Ltmp12787j# 1 main #LFT773765# 0 0

LFT773765:
  	movl	0x8(%rsi), %r12d
  	leaq	0x10(%rsi), %r9
  	movq	%r9, %rsi
LFT773765j:
  	addq	%r9, %r12
# FDATA: 1 main #LFT773765j# 1 main #Ltmp1027643# 0 0

Ltmp1027643:
  	subq	%rsi, %r12
  	cmpq	%r12, %r13
Ltmp1027643j:
  	je	Ltmp1027634
# FDATA: 1 main #Ltmp1027643j# 1 main #Ltmp1027634# 0 0
# FDATA: 1 main #Ltmp1027643j# 1 main #Ltmp1027638# 0 0

Ltmp1027638:
  	movq	data3, %rsi
  	testq	%rsi, %rsi
Ltmp1027638j:
  	je	Ltmp1027635
# FDATA: 1 main #Ltmp1027638j# 1 main #Ltmp1027635# 0 0
# FDATA: 1 main #Ltmp1027638j# 1 main #Ltmp1027636# 0 0

Ltmp1027636:
  	movl	0x8(%rsi), %r10d
  	leaq	0x10(%rsi), %r11
  	movq	%r11, %rsi
Ltmp1027636j:
  	addq	%r11, %r10
# FDATA: 1 main #Ltmp1027636j# 1 main #Ltmp1027637# 0 0

Ltmp1027637:
  	subq	%rsi, %r10
  	cmpq	%r10, %r13
Ltmp1027637j:
  	jne	Ltmp12784
# FDATA: 1 main #Ltmp1027637j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027637j# 1 main #LFT773860# 0 0

LFT773860:
  	movq	%r14, %rdi
  	callq	func2
  	testl	%eax, %eax
  	setne	%sil
  	movzbl	%sil, %eax
LFT773860j:
  	jmp	Ltmp1027626
# FDATA: 1 main #LFT773860j# 1 main #Ltmp1027626# 0 0

Ltmp12788:
  	movq	data4, %rsi
  	testq	%rsi, %rsi
Ltmp12788j:
  	jne	Ltmp1027636
# FDATA: 1 main #Ltmp12788j# 1 main #Ltmp1027636# 0 0
# FDATA: 1 main #Ltmp12788j# 1 main #Ltmp1027635# 0 0

Ltmp1027635:
  	xorl	%r10d, %r10d
Ltmp1027635j:
  	jmp	Ltmp1027637
# FDATA: 1 main #Ltmp1027635j# 1 main #Ltmp1027637# 0 0

Ltmp1027634:
  	movq	%r14, %rdi
  	callq	func2
  	testl	%eax, %eax
Ltmp1027634j:
  	je	Ltmp12775
Ltmp1027634j2:
  	jmp	Ltmp1027638
# FDATA: 1 main #Ltmp1027634j# 1 main #Ltmp12775# 0 0
# FDATA: 1 main #Ltmp1027634j2# 1 main #Ltmp1027638# 0 0

Ltmp12790:
  	movq	%r14, %rdi
  	movq	%r12, %rsi
  	callq	"func1"
  	testb	%al, %al
Ltmp12790j:
  	je	Ltmp12784
# FDATA: 1 main #Ltmp12790j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp12790j# 1 main #Ltmp1027628# 0 0

Ltmp1027628:
  	movl	$0x6, %eax
Ltmp1027628j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp1027628j# 1 main #Ltmp1027626# 0 0

Ltmp12789:
  	movq	data4, %rsi
  	testq	%rsi, %rsi
Ltmp12789j:
  	je	Ltmp1027639
# FDATA: 1 main #Ltmp12789j# 1 main #Ltmp1027639# 0 0
# FDATA: 1 main #Ltmp12789j# 1 main #LFT774000# 0 0

LFT774000:
  	movl	0x8(%rsi), %eax
  	leaq	0x10(%rsi), %rdi
  	movq	%rdi, %rsi
LFT774000j:
  	addq	%rdi, %rax
# FDATA: 1 main #LFT774000j# 1 main #Ltmp1027642# 0 0

Ltmp1027642:
  	subq	%rsi, %rax
  	cmpq	%rax, %r13
Ltmp1027642j:
  	je	Ltmp1027640
# FDATA: 1 main #Ltmp1027642j# 1 main #Ltmp1027640# 0 0
# FDATA: 1 main #Ltmp1027642j# 1 main #Ltmp1027646# 0 0

Ltmp1027646:
  	movq	data3, %rsi
  	testq	%rsi, %rsi
Ltmp1027646j:
  	je	Ltmp1027641
# FDATA: 1 main #Ltmp1027646j# 1 main #Ltmp1027641# 0 0
# FDATA: 1 main #Ltmp1027646j# 1 main #LFT774007# 0 0

LFT774007:
  	movl	0x8(%rsi), %r8d
  	leaq	0x10(%rsi), %rcx
  	movq	%rcx, %rsi
LFT774007j:
  	addq	%rcx, %r8
# FDATA: 1 main #LFT774007j# 1 main #Ltmp1027647# 0 0

Ltmp1027647:
  	subq	%rsi, %r8
  	cmpq	%r8, %r13
Ltmp1027647j:
  	jne	Ltmp12784
# FDATA: 1 main #Ltmp1027647j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027647j# 1 main #LFT774061# 0 0

LFT774061:
  	movq	%r14, %rdi
  	callq	func2
  	testl	%eax, %eax
LFT774061j:
  	jne	Ltmp12784
# FDATA: 1 main #LFT774061j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #LFT774061j# 1 main #Ltmp1027645# 0 0

Ltmp1027645:
  	cmpb	$0x0, data2
  	movl	$0x5, %eax
Ltmp1027645j:
  	je	Ltmp1027626
# FDATA: 1 main #Ltmp1027645j# 1 main #Ltmp1027626# 0 0
# FDATA: 1 main #Ltmp1027645j# 1 main #Ltmp1027650# 0 0

Ltmp1027650:
  	movl	$0x4, %eax
Ltmp1027650j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp1027650j# 1 main #Ltmp1027626# 0 0

Ltmp1027639:
  	xorl	%eax, %eax
Ltmp1027639j:
  	jmp	Ltmp1027642
# FDATA: 1 main #Ltmp1027639j# 1 main #Ltmp1027642# 0 0

Ltmp1027633:
  	xorl	%r12d, %r12d
Ltmp1027633j:
  	jmp	Ltmp1027643
# FDATA: 1 main #Ltmp1027633j# 1 main #Ltmp1027643# 0 0

Ltmp1027631:
  	xorl	%r9d, %r9d
Ltmp1027631j:
  	jmp	Ltmp1027644
# FDATA: 1 main #Ltmp1027631j# 1 main #Ltmp1027644# 0 0

Ltmp1027640:
  	movq	%r14, %rdi
  	callq	func2
  	testl	%eax, %eax
Ltmp1027640j:
  	je	Ltmp1027645
Ltmp1027640j2:
  	jmp	Ltmp1027646
# FDATA: 1 main #Ltmp1027640j# 1 main #Ltmp1027645# 0 0
# FDATA: 1 main #Ltmp1027640j2# 1 main #Ltmp1027646# 0 0

Ltmp1027641:
  	xorl	%r8d, %r8d
Ltmp1027641j:
  	jmp	Ltmp1027647
# FDATA: 1 main #Ltmp1027641j# 1 main #Ltmp1027647# 0 0

Ltmp1027629:
  	andl	$-0x4, %ebx
  	cmpb	$-0x8, %bl
Ltmp1027629j:
  	je	Ltmp12775
Ltmp1027629j2:
  	jmp	Ltmp12784
# FDATA: 1 main #Ltmp1027629j# 1 main #Ltmp12775# 0 0
# FDATA: 1 main #Ltmp1027629j2# 1 main #Ltmp12784# 0 0

Ltmp12777:
  	cmpb	$-0xa, %dil
Ltmp12777j:
  	jl	Ltmp1027648
# FDATA: 1 main #Ltmp12777j# 1 main #Ltmp1027648# 0 0
# FDATA: 1 main #Ltmp12777j# 1 main #LFT774116# 0 0

LFT774116:
  	cmpb	$-0x3, %dil
LFT774116j:
  	ja	Ltmp1027648
# FDATA: 1 main #LFT774116j# 1 main #Ltmp1027648# 0 0
# FDATA: 1 main #LFT774116j# 1 main #LFT774121# 0 0

LFT774121:
  	movl	%edi, %ecx
  	movl	$0x1, %r8d
  	andl	$-0x2, %ecx
  	cmpb	$-0x6, %cl
  	sete	%dil
  	cmpb	$0xe, %bl
LFT774121j:
  	ja	Ltmp1027649
# FDATA: 1 main #LFT774121j# 1 main #Ltmp1027649# 0 0
# FDATA: 1 main #LFT774121j# 1 main #LFT774198# 0 0

LFT774198:
  	movl	$0x4e08, %r8d
  	movl	%ebx, %ecx
  	shrq	%cl, %r8
  	notq	%r8
LFT774198j:
  	andl	$0x1, %r8d
# FDATA: 1 main #LFT774198j# 1 main #Ltmp1027649# 0 0

Ltmp1027649:
  	testb	%r8b, %r8b
Ltmp1027649j:
  	je	Ltmp1027648
# FDATA: 1 main #Ltmp1027649j# 1 main #Ltmp1027648# 0 0
# FDATA: 1 main #Ltmp1027649j# 1 main #LFT774233# 0 0

LFT774233:
  	movl	$0x1, %eax
  	testb	%dil, %dil
LFT774233j:
  	je	Ltmp1027626
# FDATA: 1 main #LFT774233j# 1 main #Ltmp1027626# 0 0
# FDATA: 1 main #LFT774233j# 1 main #Ltmp1027648# 0 0

Ltmp1027648:
  	movl	$0x3, %eax
Ltmp1027648j:
  	jmp	Ltmp1027626
# FDATA: 1 main #Ltmp1027648j# 1 main #Ltmp1027626# 0 0

Ltmp1027621:
  	cmpb	$0x0, data2
  	movl	$0x5, %eax
Ltmp1027621j:
  	je	Ltmp1027626
Ltmp1027621j2:
  	jmp	Ltmp1027650
# FDATA: 1 main #Ltmp1027621j# 1 main #Ltmp1027626# 0 0
# FDATA: 1 main #Ltmp1027621j2# 1 main #Ltmp1027650# 0 0

Ltmp12781:
  	cmpb	$0xe, %dil
Ltmp12781j:
  	je	Ltmp1027651
# FDATA: 1 main #Ltmp12781j# 1 main #Ltmp1027651# 0 0
# FDATA: 1 main #Ltmp12781j# 1 main #LFT774325# 0 0

LFT774325:
  	andl	$-0x4, %ebx
  	cmpb	$-0x8, %bl
  	setne	%dl
LFT774325j:
  	jmp	Ltmp1027627
# FDATA: 1 main #LFT774325j# 1 main #Ltmp1027627# 0 0

Ltmp12782:
  	cmpb	$0xe, %dil
Ltmp12782j:
  	je	Ltmp1027628
# FDATA: 1 main #Ltmp12782j# 1 main #Ltmp1027628# 0 0
# FDATA: 1 main #Ltmp12782j# 1 main #LFT774343# 0 0

LFT774343:
  	xorl	%eax, %eax
  	cmpb	$-0x4, %dil
  	setge	%al
LFT774343j:
  	jmp	Ltmp1027626
# FDATA: 1 main #LFT774343j# 1 main #Ltmp1027626# 0 0

Ltmp1027622:
  	cmpw	$0xfb, %si
Ltmp1027622j:
  	jne	Ltmp1027652
# FDATA: 1 main #Ltmp1027622j# 1 main #Ltmp1027652# 0 0
# FDATA: 1 main #Ltmp1027622j# 1 main #Ltmp1027651# 0 0

Ltmp1027651:
  	cmpb	$0x0, data1
Ltmp1027651j:
  	je	Ltmp12784
Ltmp1027651j2:
  	jmp	Ltmp1027628
# FDATA: 1 main #Ltmp1027651j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027651j2# 1 main #Ltmp1027628# 0 0

Ltmp1027623:
  	cmpb	$-0x3, %bl
Ltmp1027623j:
  	jne	Ltmp12784
# FDATA: 1 main #Ltmp1027623j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027623j# 1 main #LFT774382# 0 0

LFT774382:
  	movl	$0x7, %eax
LFT774382j:
  	jmp	Ltmp1027626
# FDATA: 1 main #LFT774382j# 1 main #Ltmp1027626# 0 0

Ltmp1027652:
  	cmpw	$0xf5, %si
Ltmp1027652j:
  	je	Ltmp1027653
# FDATA: 1 main #Ltmp1027652j# 1 main #Ltmp1027653# 0 0
# FDATA: 1 main #Ltmp1027652j# 1 main #LFT774387# 0 0

LFT774387:
  	cmpw	$0xa00, %si
LFT774387j:
  	je	Ltmp1027628
# FDATA: 1 main #LFT774387j# 1 main #Ltmp1027628# 0 0
# FDATA: 1 main #LFT774387j# 1 main #LFT774414# 0 0

LFT774414:
  	cmpw	$0xfd, %si
LFT774414j:
  	je	Ltmp12784
# FDATA: 1 main #LFT774414j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #LFT774414j# 1 main #LFT774417# 0 0

LFT774417:
  	cmpw	$0x3, %si
LFT774417j:
  	je	Ltmp1027654
# FDATA: 1 main #LFT774417j# 1 main #Ltmp1027654# 0 0
# FDATA: 1 main #LFT774417j# 1 main #LFT774505# 0 0

LFT774505:
  	cmpb	$0xe, %sil
LFT774505j:
  	je	Ltmp12775
LFT774505j2:
  	jmp	Ltmp12784
# FDATA: 1 main #LFT774505j# 1 main #Ltmp12775# 0 0
# FDATA: 1 main #LFT774505j2# 1 main #Ltmp12784# 0 0

Ltmp1027653:
  	cmpb	$0x0, data1
Ltmp1027653j:
  	jne	Ltmp12784
Ltmp1027653j2:
  	jmp	Ltmp1027628
# FDATA: 1 main #Ltmp1027653j# 1 main #Ltmp12784# 0 0
# FDATA: 1 main #Ltmp1027653j2# 1 main #Ltmp1027628# 0 0
.size main, .-main

.globl func1
.type func1, %function
func1:
  ret
.size func1, .-func1

.globl func2
.type func2, %function
func2:
  ret
.size func2, .-func2

  .data
data1: .asciz "data001"
data2: .asciz "data002"
data3: .asciz "data003"
data4: .asciz "data004"

  .section .rodata
  .globl JUMP_TABLE0
JUMP_TABLE0:
  .quad Ltmp12774
  .quad Ltmp12775
  .quad Ltmp12776
  .quad Ltmp12776
  .quad Ltmp12777
  .quad Ltmp12778
  .quad Ltmp12779
  .quad Ltmp12776
  .quad Ltmp12780
  .quad Ltmp12781
  .quad Ltmp12782
  .quad Ltmp12783
  .quad Ltmp12784
  .quad Ltmp12784

  .globl JUMP_TABLE1
JUMP_TABLE1:
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12785
  .quad Ltmp12786
  .quad Ltmp12784
  .quad Ltmp12787
  .quad Ltmp12787
  .quad Ltmp12784
  .quad Ltmp12786
  .quad Ltmp12784
  .quad Ltmp12786
  .quad Ltmp12784
  .quad Ltmp12784
  .quad Ltmp12788
  .quad Ltmp12786
  .quad Ltmp12788
  .quad Ltmp12784
  .quad Ltmp12784
  .quad Ltmp12784
  .quad Ltmp12789
  .quad Ltmp12786
  .quad Ltmp12790


