; RUN: llc < %s -march=x86 > %t
; RUN: grep {fld	%%st(0)} %t
; PR4459

declare x86_fp80 @ceil(x86_fp80)

declare void @test(x86_fp80)

define void @test2(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @ceil(x86_fp80 %a)
	call void asm sideeffect "fistpl $0", "{st}"( x86_fp80 %0)
	call void @test(x86_fp80 %0 )
        ret void
}
