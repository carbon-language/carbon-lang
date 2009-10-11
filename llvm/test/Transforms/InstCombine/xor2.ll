; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1253
define i1 @test0(i32 %A) {
; CHECK: @test0
; CHECK: %C = icmp slt i32 %A, 0
	%B = xor i32 %A, -2147483648
	%C = icmp sgt i32 %B, -1
	ret i1 %C
}

define i1 @test1(i32 %A) {
; CHECK: @test1
; CHECK: %C = icmp slt i32 %A, 0
	%B = xor i32 %A, 12345
	%C = icmp slt i32 %B, 0
	ret i1 %C
}

; PR1014
define i32 @test2(i32 %tmp1) {
; CHECK:      @test2
; CHECK-NEXT:   or i32 %tmp1, 8 
; CHECK-NEXT:   and i32
; CHECK-NEXT:   ret i32
        %ovm = and i32 %tmp1, 32
        %ov3 = add i32 %ovm, 145
        %ov110 = xor i32 %ov3, 153
        ret i32 %ov110
}
