// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

# CHECK: callq a
        callq a

# CHECK: leaq	-40(%rbp), %r15
	leaq	-40(%rbp), %r15



// rdar://8013734 - Alias dr6=db6
mov %dr6, %rax
mov %db6, %rax
# CHECK: movq	%dr6, %rax
# CHECK: movq	%dr6, %rax
