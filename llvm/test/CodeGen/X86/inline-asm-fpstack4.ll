; RUN: llc < %s -march=x86 | FileCheck %s
; PR4484

; ceil leaves a value on the stack that is needed after the asm.
; CHECK: ceil
; CHECK-NOT: fstp
; Load %a from stack after ceil
; CHECK: fldt
; CHECK-NOT: fxch
; CHECK: fistpl
; CHECK-NOT: fstp
; Set up call to test.
; CHECK: fstpt
; CHECK: test
define void @test2(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @ceil()
	call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %a)
	call void @test(x86_fp80 %0)
	ret void
}

declare x86_fp80 @ceil()
declare void @test(x86_fp80)
