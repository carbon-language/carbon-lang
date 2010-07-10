; RUN: llc < %s -march=x86 | FileCheck %s
; PR4185

; Passing a non-killed value to asm in {st}.
; Make sure it is duped before.
; asm kills st(0), so we shouldn't pop anything
; CHECK: fld %st(0)
; CHECK: fistpl
; CHECK-NOT: fstp
; CHECK: fistpl
; CHECK-NOT: fstp
; CHECK: ret
define void @test() {
return:
	call void asm sideeffect "fistpl $0", "{st}"(double 1.000000e+06)
	call void asm sideeffect "fistpl $0", "{st}"(double 1.000000e+06)
	ret void
}

; A valid alternative would be to remat the constant pool load before each
; inline asm.
