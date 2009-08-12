; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
entry:
; CHECK: f1:
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
	%tmp = sub i64 %a, %b
	ret i64 %tmp
}

define i64 @f2(i64 %a, i64 %b) {
entry:
; CHECK: f2:
; CHECK: adds r0, r0, r0
; CHECK: adcs r1, r1
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
        %tmp1 = shl i64 %a, 1
	%tmp2 = sub i64 %tmp1, %b
	ret i64 %tmp2
}
