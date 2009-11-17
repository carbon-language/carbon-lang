; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @test1(i32 %x) {
; CHECK: test1
; CHECK: uxtb16.w  r0, r0
	%tmp1 = and i32 %x, 16711935		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32 @test2(i32 %x) {
; CHECK: test2
; CHECK: uxtb16.w  r0, r0, ror #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test3(i32 %x) {
; CHECK: test3
; CHECK: uxtb16.w  r0, r0, ror #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test4(i32 %x) {
; CHECK: test4
; CHECK: uxtb16.w  r0, r0, ror #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp6 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test5(i32 %x) {
; CHECK: test5
; CHECK: uxtb16.w  r0, r0, ror #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test6(i32 %x) {
; CHECK: test6
; CHECK: uxtb16.w  r0, r0, ror #16
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test7(i32 %x) {
; CHECK: test7
; CHECK: uxtb16.w  r0, r0, ror #16
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test8(i32 %x) {
; CHECK: test8
; CHECK: uxtb16.w  r0, r0, ror #24
	%tmp1 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711680		; <i32> [#uses=1]
	%tmp5 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test9(i32 %x) {
; CHECK: test9
; CHECK: uxtb16.w  r0, r0, ror #24
	%tmp1 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp5, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test10(i32 %p0) {
; CHECK: test10
; CHECK: mov.w r1, #16253176
; CHECK: and.w r0, r1, r0, lsr #7
; CHECK: lsrs  r1, r0, #5
; CHECK: uxtb16.w  r1, r1
; CHECK: orr.w r0, r1, r0

	%tmp1 = lshr i32 %p0, 7		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16253176		; <i32> [#uses=2]
	%tmp4 = lshr i32 %tmp2, 5		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 458759		; <i32> [#uses=1]
	%tmp7 = or i32 %tmp5, %tmp2		; <i32> [#uses=1]
	ret i32 %tmp7
}
