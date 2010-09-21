; RUN: llc < %s -march=x86 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s --check-prefix=X64

define i32 @test1() nounwind readonly {
entry:
	%tmp = load i32* addrspace(256)* getelementptr (i32* addrspace(256)* inttoptr (i32 72 to i32* addrspace(256)*), i32 31)		; <i32*> [#uses=1]
	%tmp1 = load i32* %tmp		; <i32> [#uses=1]
	ret i32 %tmp1
}
; X32: test1:
; X32: 	movl	%gs:196, %eax
; X32: 	movl	(%eax), %eax
; X32: 	ret

; X64: test1:
; X64: 	movq	%gs:320, %rax
; X64: 	movl	(%rax), %eax
; X64: 	ret

define i64 @test2(void (i8*)* addrspace(256)* %tmp8) nounwind {
entry:
  %tmp9 = load void (i8*)* addrspace(256)* %tmp8, align 8
  tail call void %tmp9(i8* undef) nounwind optsize
  ret i64 0
}

; rdar://8453210
; X32: test2:
; X32: movl	%gs:(%eax), %eax
; X32: movl	%eax, (%esp)
; X32: call	*%eax

; X64: test2:
; X64: movq	%gs:(%rdi), %rax
; X64: callq	*%rax
