; RUN: opt < %s -reassociate -S | FileCheck %s

define i32 @test1(i32 %a, i32 %b) {
; CHECK: @test1
; CHECK: mul i32 %a, %a
; CHECK-NEXT: mul i32 %a, 2
; CHECK-NEXT: add
; CHECK-NEXT: mul
; CHECK-NEXT: add
; CHECK-NEXT: ret

entry:
	%tmp.2 = mul i32 %a, %a
	%tmp.5 = shl i32 %a, 1
	%tmp.6 = mul i32 %tmp.5, %b
	%tmp.10 = mul i32 %b, %b
	%tmp.7 = add i32 %tmp.6, %tmp.2
	%tmp.11 = add i32 %tmp.7, %tmp.10
	ret i32 %tmp.11
}

define i32 @test2(i32 %t) {
; CHECK: @test2
; CHECK: mul
; CHECK-NEXT: add
; CHECK-NEXT: ret

entry:
	%a = mul i32 %t, 6
	%b = mul i32 %t, 36
	%c = add i32 %b, 15
	%d = add i32 %c, %a
	ret i32 %d
}

