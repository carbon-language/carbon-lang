; RUN: llc < %s -march=x86 -asm-verbose | FileCheck %s
; Check that register copies in the landing pad come after the EH_LABEL

declare i32 @f()

define i32 @phi(i32 %x) {
entry:
	%a = invoke i32 @f()
			to label %cont unwind label %lpad		; <i32> [#uses=1]

cont:		; preds = %entry
	%b = invoke i32 @f()
			to label %cont2 unwind label %lpad		; <i32> [#uses=1]

cont2:		; preds = %cont
	ret i32 %b

lpad:		; preds = %cont, %entry
	%v = phi i32 [ %x, %entry ], [ %a, %cont ]		; <i32> [#uses=1]
	ret i32 %v
}

; CHECK: lpad
; CHECK-NEXT: Ltmp
