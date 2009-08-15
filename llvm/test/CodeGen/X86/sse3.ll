; These are tests for SSE3 codegen.  Yonah has SSE3 and earlier but not SSSE3+.

; RUN: llvm-as < %s | llc -march=x86-64 -mcpu=yonah | FileCheck %s --check-prefix=X64

; Test for v8xi16 lowering where we extract the first element of the vector and
; placed it in the second element of the result.

define void @shuf1(<8 x i16>* %dest, <8 x i16>* %old) nounwind {
entry:
	%tmp3 = load <8 x i16>* %old
	%tmp6 = shufflevector <8 x i16> %tmp3,
                <8 x i16> < i16 0, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef >,
                <8 x i32> < i32 8, i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef  >
	store <8 x i16> %tmp6, <8 x i16>* %dest
	ret void
        
; X64: shuf1:
; X64: 	movddup	(%rsi), %xmm0
; X64:  pshuflw	$0, %xmm0, %xmm0
; X64:	xorl	%eax, %eax
; X64:	pinsrw	$0, %eax, %xmm0
; X64:	movaps	%xmm0, (%rdi)
; X64:	ret
}