; RUN: llc < %s -march=xcore | FileCheck %s
define i64 @add64(i64 %a, i64 %b) {
	%result = add i64 %a, %b
	ret i64 %result
}
; CHECK: add64
; CHECK: ldc r11, 0
; CHECK-NEXT: ladd r2, r0, r0, r2, r11
; CHECK-NEXT: ladd r2, r1, r1, r3, r2
; CHECK-NEXT: retsp 0

define i64 @sub64(i64 %a, i64 %b) {
	%result = sub i64 %a, %b
	ret i64 %result
}
; CHECK: sub64
; CHECK: ldc r11, 0
; CHECK-NEXT: lsub r2, r0, r0, r2, r11
; CHECK-NEXT: lsub r2, r1, r1, r3, r2
; CHECK-NEXT: retsp 0
