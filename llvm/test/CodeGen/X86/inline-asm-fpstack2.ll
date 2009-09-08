; RUN: llc < %s -march=x86 > %t
; RUN: grep {fld	%%st(0)} %t
; PR4185

define void @test() {
return:
	call void asm sideeffect "fistpl $0", "{st}"(double 1.000000e+06)
	call void asm sideeffect "fistpl $0", "{st}"(double 1.000000e+06)
	ret void
}
