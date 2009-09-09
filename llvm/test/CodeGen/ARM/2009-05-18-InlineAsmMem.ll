; RUN: llc < %s -march=arm | grep swp
; PR4091

define void @foo(i32 %i, i32* %p) nounwind {
	%asmtmp = call i32 asm sideeffect "swp $0, $2, $3", "=&r,=*m,r,*m,~{memory}"(i32* %p, i32 %i, i32* %p) nounwind
	ret void
}
