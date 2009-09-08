; RUN: llc < %s -march=x86-64           | grep {movq.*(%rsi), %rax}
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep {movsd.*(%eax),}

; Uses movsd to load / store i64 values if sse2 is available.

; rdar://6659858

define void @foo(i64* %x, i64* %y) nounwind  {
entry:
	%tmp1 = load i64* %y, align 8		; <i64> [#uses=1]
	store i64 %tmp1, i64* %x, align 8
	ret void
}
