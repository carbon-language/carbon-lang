; RUN: llvm-as < %s | llc -march=x86-64
; rdar://7066579

	type { i64, i64, i64, i64, i64 }		; type %0

define void @t() nounwind {
entry:
	%asmtmp = call %0 asm sideeffect "mov    %cr0, $0       \0Amov    %cr2, $1       \0Amov    %cr3, $2       \0Amov    %cr4, $3       \0Amov    %cr8, $0       \0A", "=q,=q,=q,=q,=q,~{dirflag},~{fpsr},~{flags}"() nounwind		; <%0> [#uses=0]
	ret void
}
