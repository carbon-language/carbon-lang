; RUN: llc < %s -march=arm | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1:
; CHECK: subs r
; CHECK: sbc r
entry:
	%tmp = sub i64 %a, %b
	ret i64 %tmp
}

define i64 @f2(i64 %a, i64 %b) {
; CHECK: f2:
; CHECK: adc r
; CHECK: subs r
; CHECK: sbc r
entry:
        %tmp1 = shl i64 %a, 1
	%tmp2 = sub i64 %tmp1, %b
	ret i64 %tmp2
}

; add with live carry
define i64 @f3(i32 %al, i32 %bl) {
; CHECK: f3:
; CHECK: adds r
; CHECK: adcs r
; CHECK: adc r
entry:
        ; unsigned wide add
        %aw = zext i32 %al to i64
        %bw = zext i32 %bl to i64
        %cw = add i64 %aw, %bw
        ; ch == carry bit
        %ch = lshr i64 %cw, 32
	%dw = add i64 %ch, %bw
	ret i64 %dw
}
