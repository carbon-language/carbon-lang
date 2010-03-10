; RUN: llc < %s -march=xcore | FileCheck %s
define i64 @umul_lohi(i32 %a, i32 %b) {
entry:
	%0 = zext i32 %a to i64
	%1 = zext i32 %b to i64
	%2 = mul i64 %1, %0
	ret i64 %2
}
; CHECK: umul_lohi:
; CHECK: ldc r2, 0
; CHECK-NEXT: lmul r1, r0, r1, r0, r2, r2
; CHECK-NEXT: retsp 0

define i64 @smul_lohi(i32 %a, i32 %b) {
entry:
	%0 = sext i32 %a to i64
	%1 = sext i32 %b to i64
	%2 = mul i64 %1, %0
	ret i64 %2
}
; CHECK: smul_lohi:
; CHECK: ldc r2, 0
; CHECK-NEXT: mov r3, r2
; CHECK-NEXT: maccs r2, r3, r1, r0
; CHECK-NEXT: mov r0, r3
; CHECK-NEXT: mov r1, r2
; CHECK-NEXT: retsp 0

define i64 @mul64(i64 %a, i64 %b) {
entry:
	%0 = mul i64 %a, %b
	ret i64 %0
}
; CHECK: mul64:
; CHECK: ldc r11, 0
; CHECK-NEXT: lmul r11, r4, r0, r2, r11, r11
; CHECK-NEXT: mul r0, r0, r3
; CHECK-NEXT: lmul r0, r1, r1, r2, r11, r0
; CHECK-NEXT: mov r0, r4
