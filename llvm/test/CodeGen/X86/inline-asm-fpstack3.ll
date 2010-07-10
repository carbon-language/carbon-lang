; RUN: llc < %s -march=x86 | FileCheck %s
; PR4459

; The return value from ceil must be duped before being consumed by asm.
; CHECK: ceil
; CHECK: fld %st(0)
; CHECK-NOT: fxch
; CHECK: fistpl
; CHECK-NOT: fxch
; CHECK: fstpt
; CHECK: test
define void @test2(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @ceil(x86_fp80 %a)
	call void asm sideeffect "fistpl $0", "{st}"( x86_fp80 %0)
	call void @test(x86_fp80 %0 )
        ret void
}
declare x86_fp80 @ceil(x86_fp80)
declare void @test(x86_fp80)
