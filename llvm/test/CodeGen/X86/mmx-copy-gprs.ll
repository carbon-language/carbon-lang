; RUN: llvm-as < %s | llc -march=x86-64           | grep {movq.*(%rsi), %rax}
; RUN: llvm-as < %s | llc -march=x86 -mattr=-sse2 | grep {movl.*4(%eax),}
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {movsd.(%eax),}

; This test should use GPRs to copy the mmx value, not MMX regs.  Using mmx regs,
; increases the places that need to use emms.

; rdar://5741668

define void @foo(<1 x i64>* %x, <1 x i64>* %y) nounwind  {
entry:
	%tmp1 = load <1 x i64>* %y, align 8		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp1, <1 x i64>* %x, align 8
	ret void
}
