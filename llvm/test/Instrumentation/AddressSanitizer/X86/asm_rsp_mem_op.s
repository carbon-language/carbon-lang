# The test verifies that memory references through %rsp are correctly
# adjusted after instrumentation.

# RUN: llvm-mc %s -triple=x86_64-unknown-linux-gnu -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

# CHECK-LABEL: rsp_access
# CHECK: leaq -128(%rsp), %rsp
# CHECK: pushq %rax
# CHECK: pushq %rdi
# CHECK: pushfq
# CHECK: leaq 160(%rsp), %rdi
# CHECK: callq __asan_report_load8@PLT
# CHECK: popfq
# CHECK: popq %rdi
# CHECK: popq %rax
# CHECK: leaq 128(%rsp), %rsp
# CHECK: movq 8(%rsp), %rax
# CHECK: retq

	.text
	.globl rsp_access
	.type rsp_access,@function
rsp_access:
	movq 8(%rsp), %rax
	retq

# CHECK-LABEL: rsp_32bit_access
# CHECK: leaq -128(%rsp), %rsp
# CHECK: pushq %rax
# CHECK: pushq %rdi
# CHECK: pushfq
# CHECK: leaq 2147483647(%rsp), %rdi
# CHECK: leaq 145(%rdi), %rdi
# CHECK: callq __asan_report_load8@PLT
# CHECK: popfq
# CHECK: popq %rdi
# CHECK: popq %rax
# CHECK: leaq 128(%rsp), %rsp
# CHECK: movq 2147483640(%rsp), %rax
# CHECK: retq
	.globl rsp_32bit_access
	.type rsp_32bit_access,@function
rsp_32bit_access:
	movq 2147483640(%rsp), %rax
	retq
