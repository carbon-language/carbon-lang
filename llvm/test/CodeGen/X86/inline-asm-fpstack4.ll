; RUN: llc < %s -march=x86
; PR4484

declare x86_fp80 @ceil()

declare void @test(x86_fp80)

define void @test2(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @ceil()
	call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %a)
	call void @test(x86_fp80 %0)
	ret void
}

