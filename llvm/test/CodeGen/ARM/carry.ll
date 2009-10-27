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
