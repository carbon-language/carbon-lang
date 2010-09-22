; RUN: llc < %s -march=x86 -mattr=sse41 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -march=x86-64 -mattr=sse41 | FileCheck %s --check-prefix=X64

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
; X32: movl	{{.*}}(%esp), %eax
; X32: calll	*%gs:(%eax)

; X64: test2:
; X64: callq	*%gs:(%rdi)




define <2 x i64> @pmovsxwd_1(i64 addrspace(256)* %p) nounwind readonly {
entry:
  %0 = load i64 addrspace(256)* %p
  %tmp2 = insertelement <2 x i64> zeroinitializer, i64 %0, i32 0
  %1 = bitcast <2 x i64> %tmp2 to <8 x i16>
  %2 = tail call <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16> %1) nounwind readnone
  %3 = bitcast <4 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
  
; X32: pmovsxwd_1:
; X32: 	movl	4(%esp), %eax
; X32: 	pmovsxwd	%gs:(%eax), %xmm0
; X32: 	ret

; X64: pmovsxwd_1:
; X64:	pmovsxwd	%gs:(%rdi), %xmm0
; X64:	ret
}

declare <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16>) nounwind readnone
