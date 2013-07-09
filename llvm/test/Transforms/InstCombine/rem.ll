; This test makes sure that urem instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; END.

define i32 @test1(i32 %A) {
; CHECK: @test1
; CHECK-NEXT: ret i32 0
	%B = srem i32 %A, 1	; ISA constant 0
	ret i32 %B
}

define i32 @test2(i32 %A) {	; 0 % X = 0, we don't need to preserve traps
; CHECK: @test2
; CHECK-NEXT: ret i32 0
	%B = srem i32 0, %A
	ret i32 %B
}

define i32 @test3(i32 %A) {
; CHECK: @test3
; CHECK-NEXT: [[AND:%.*]] = and i32 %A, 7
; CHECK-NEXT: ret i32 [[AND]]
	%B = urem i32 %A, 8
	ret i32 %B
}

define i1 @test3a(i32 %A) {
; CHECK: @test3a
; CHECK-NEXT: [[AND:%.*]] = and i32 %A, 7
; CHECK-NEXT: [[CMP:%.*]] = icmp ne i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
	%B = srem i32 %A, -8
	%C = icmp ne i32 %B, 0
	ret i1 %C
}

define i32 @test4(i32 %X, i1 %C) {
; CHECK: @test4
; CHECK-NEXT: [[SEL:%.*]] = select i1 %C, i32 0, i32 7
; CHECK-NEXT: [[AND:%.*]] = and i32 [[SEL]], %X
	%V = select i1 %C, i32 1, i32 8
	%R = urem i32 %X, %V
	ret i32 %R
}

define i32 @test5(i32 %X, i8 %B) {
; CHECK: @test5
; CHECK-NEXT: [[ZEXT:%.*]] = zext i8 %B to i32
; CHECK-NEXT: [[SHL:%.*]] = shl nuw i32 32, [[ZEXT]]
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SHL]], -1
; CHECK-NEXT: [[AND:%.*]] = and i32 [[ADD]], %X
; CHECK-NEXT: ret i32 [[AND]]
	%shift.upgrd.1 = zext i8 %B to i32
	%Amt = shl i32 32, %shift.upgrd.1
	%V = urem i32 %X, %Amt
	ret i32 %V
}

define i32 @test6(i32 %A) {
; CHECK: @test6
; CHECK-NEXT: ret i32 undef
	%B = srem i32 %A, 0	;; undef
	ret i32 %B
}

define i32 @test7(i32 %A) {
; CHECK: @test7
; CHECK-NEXT: ret i32 0
	%B = mul i32 %A, 8
	%C = srem i32 %B, 4
	ret i32 %C
}

define i32 @test8(i32 %A) {
; CHECK: @test8
; CHECK-NEXT: ret i32 0
	%B = shl i32 %A, 4
	%C = srem i32 %B, 8
	ret i32 %C
}

define i32 @test9(i32 %A) {
; CHECK: @test9
; CHECK-NEXT: ret i32 0
	%B = mul i32 %A, 64
	%C = urem i32 %B, 32
	ret i32 %C
}

define i32 @test10(i8 %c) {
; CHECK: @test10
; CHECK-NEXT: ret i32 0
	%tmp.1 = zext i8 %c to i32
	%tmp.2 = mul i32 %tmp.1, 4
	%tmp.3 = sext i32 %tmp.2 to i64
	%tmp.5 = urem i64 %tmp.3, 4
	%tmp.6 = trunc i64 %tmp.5 to i32
	ret i32 %tmp.6
}

define i32 @test11(i32 %i) {
; CHECK: @test11
; CHECK-NEXT: ret i32 0
	%tmp.1 = and i32 %i, -2
	%tmp.3 = mul i32 %tmp.1, 2
	%tmp.5 = urem i32 %tmp.3, 4
	ret i32 %tmp.5
}

define i32 @test12(i32 %i) {
; CHECK: @test12
; CHECK-NEXT: ret i32 0
	%tmp.1 = and i32 %i, -4
	%tmp.5 = srem i32 %tmp.1, 2
	ret i32 %tmp.5
}

define i32 @test13(i32 %i) {
; CHECK: @test13
; CHECK-NEXT: ret i32 0
	%x = srem i32 %i, %i
	ret i32 %x
}

define i64 @test14(i64 %x, i32 %y) {
; CHECK: @test14
; CHECK-NEXT: [[SHL:%.*]] = shl i32 1, %y
; CHECK-NEXT: [[ZEXT:%.*]] = zext i32 [[SHL]] to i64
; CHECK-NEXT: [[ADD:%.*]] = add i64 [[ZEXT]], -1
; CHECK-NEXT: [[AND:%.*]] = and i64 [[ADD]], %x
; CHECK-NEXT: ret i64 [[AND]]
	%shl = shl i32 1, %y
	%zext = zext i32 %shl to i64
	%urem = urem i64 %x, %zext
	ret i64 %urem
}

define i64 @test15(i32 %x, i32 %y) {
; CHECK: @test15
; CHECK-NEXT: [[SHL:%.*]] = shl nuw i32 1, %y
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SHL]], -1
; CHECK-NEXT: [[AND:%.*]] = and i32 [[ADD]], %x
; CHECK-NEXT: [[ZEXT:%.*]] = zext i32 [[AND]] to i64
; CHECK-NEXT: ret i64 [[ZEXT]]
	%shl = shl i32 1, %y
	%zext0 = zext i32 %shl to i64
	%zext1 = zext i32 %x to i64
	%urem = urem i64 %zext1, %zext0
	ret i64 %urem
}

define i32 @test16(i32 %x, i32 %y) {
; CHECK: @test16
; CHECK-NEXT: [[SHR:%.*]] = lshr i32 %y, 11
; CHECK-NEXT: [[AND:%.*]] = and i32 [[SHR]], 4
; CHECK-NEXT: [[OR:%.*]] = or i32 [[AND]], 3
; CHECK-NEXT: [[REM:%.*]] = and i32 [[OR]], %x
; CHECK-NEXT: ret i32 [[REM]]
	%shr = lshr i32 %y, 11
	%and = and i32 %shr, 4
	%add = add i32 %and, 4
	%rem = urem i32 %x, %add
	ret i32 %rem
}
