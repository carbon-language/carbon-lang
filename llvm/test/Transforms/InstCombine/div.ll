; This test makes sure that div instructions are properly eliminated.

; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
        %B = sdiv i32 %A, 1             ; <i32> [#uses=1]
        ret i32 %B
; CHECK-LABEL: @test1(
; CHECK-NEXT: ret i32 %A
}

define i32 @test2(i32 %A) {
        ; => Shift
        %B = udiv i32 %A, 8             ; <i32> [#uses=1]
        ret i32 %B
; CHECK-LABEL: @test2(
; CHECK-NEXT: lshr i32 %A, 3
}

define i32 @test3(i32 %A) {
        ; => 0, don't need to keep traps
        %B = sdiv i32 0, %A             ; <i32> [#uses=1]
        ret i32 %B
; CHECK-LABEL: @test3(
; CHECK-NEXT: ret i32 0
}

define i32 @test4(i32 %A) {
        ; 0-A
        %B = sdiv i32 %A, -1            ; <i32> [#uses=1]
        ret i32 %B
; CHECK-LABEL: @test4(
; CHECK-NEXT: sub i32 0, %A
}

define i32 @test5(i32 %A) {
        %B = udiv i32 %A, -16           ; <i32> [#uses=1]
        %C = udiv i32 %B, -4            ; <i32> [#uses=1]
        ret i32 %C
; CHECK-LABEL: @test5(
; CHECK-NEXT: ret i32 0
}

define i1 @test6(i32 %A) {
        %B = udiv i32 %A, 123           ; <i32> [#uses=1]
        ; A < 123
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
; CHECK-LABEL: @test6(
; CHECK-NEXT: icmp ult i32 %A, 123
}

define i1 @test7(i32 %A) {
        %B = udiv i32 %A, 10            ; <i32> [#uses=1]
        ; A >= 20 && A < 30
        %C = icmp eq i32 %B, 2          ; <i1> [#uses=1]
        ret i1 %C
; CHECK-LABEL: @test7(
; CHECK-NEXT: add i32 %A, -20
; CHECK-NEXT: icmp ult i32
}

define i1 @test8(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A >= 246
        %C = icmp eq i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
; CHECK-LABEL: @test8(
; CHECK-NEXT: icmp ugt i8 %A, -11
}

define i1 @test9(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A < 246
        %C = icmp ne i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
; CHECK-LABEL: @test9(
; CHECK-NEXT: icmp ult i8 %A, -10
}

define i32 @test10(i32 %X, i1 %C) {
        %V = select i1 %C, i32 64, i32 8                ; <i32> [#uses=1]
        %R = udiv i32 %X, %V            ; <i32> [#uses=1]
        ret i32 %R
; CHECK-LABEL: @test10(
; CHECK-NEXT: select i1 %C, i32 6, i32 3
; CHECK-NEXT: lshr i32 %X
}

define i32 @test11(i32 %X, i1 %C) {
        %A = select i1 %C, i32 1024, i32 32             ; <i32> [#uses=1]
        %B = udiv i32 %X, %A            ; <i32> [#uses=1]
        ret i32 %B
; CHECK-LABEL: @test11(
; CHECK-NEXT: select i1 %C, i32 10, i32 5
; CHECK-NEXT: lshr i32 %X
}

; PR2328
define i32 @test12(i32 %x) nounwind  {
	%tmp3 = udiv i32 %x, %x		; 1
	ret i32 %tmp3
; CHECK-LABEL: @test12(
; CHECK-NEXT: ret i32 1
}

define i32 @test13(i32 %x) nounwind  {
	%tmp3 = sdiv i32 %x, %x		; 1
	ret i32 %tmp3
; CHECK-LABEL: @test13(
; CHECK-NEXT: ret i32 1
}

define i32 @test14(i8 %x) nounwind {
	%zext = zext i8 %x to i32
	%div = udiv i32 %zext, 257	; 0
	ret i32 %div
; CHECK-LABEL: @test14(
; CHECK-NEXT: ret i32 0
}

; PR9814
define i32 @test15(i32 %a, i32 %b) nounwind {
  %shl = shl i32 1, %b
  %div = lshr i32 %shl, 2
  %div2 = udiv i32 %a, %div
  ret i32 %div2
; CHECK-LABEL: @test15(
; CHECK-NEXT: add i32 %b, -2
; CHECK-NEXT: lshr i32 %a, 
; CHECK-NEXT: ret i32
}

define <2 x i64> @test16(<2 x i64> %x) nounwind {
  %shr = lshr <2 x i64> %x, <i64 5, i64 5>
  %div = udiv <2 x i64> %shr, <i64 6, i64 6>
  ret <2 x i64> %div
; CHECK-LABEL: @test16(
; CHECK-NEXT: udiv <2 x i64> %x, <i64 192, i64 192>
; CHECK-NEXT: ret <2 x i64>
}

define <2 x i64> @test17(<2 x i64> %x) nounwind {
  %neg = sub nsw <2 x i64> zeroinitializer, %x
  %div = sdiv <2 x i64> %neg, <i64 3, i64 4>
  ret <2 x i64> %div
; CHECK-LABEL: @test17(
; CHECK-NEXT: sdiv <2 x i64> %x, <i64 -3, i64 -4>
; CHECK-NEXT: ret <2 x i64>
}

define <2 x i64> @test18(<2 x i64> %x) nounwind {
  %div = sdiv <2 x i64> %x, <i64 -1, i64 -1>
  ret <2 x i64> %div
; CHECK-LABEL: @test18(
; CHECK-NEXT: sub <2 x i64> zeroinitializer, %x
; CHECK-NEXT: ret <2 x i64>
}

define i32 @test19(i32 %x) {
  %A = udiv i32 1, %x
  ret i32 %A
; CHECK-LABEL: @test19(
; CHECK-NEXT: icmp eq i32 %x, 1
; CHECK-NEXT: zext i1 %{{.*}} to i32
; CHECK-NEXT: ret i32
}

define i32 @test20(i32 %x) {
  %A = sdiv i32 1, %x
  ret i32 %A
; CHECK-LABEL: @test20(
; CHECK-NEXT: add i32 %x, 1
; CHECK-NEXT: icmp ult i32 %{{.*}}, 3
; CHECK-NEXT: select i1 %{{.*}}, i32 %x, i32 {{.*}}
; CHECK-NEXT: ret i32
}

define i32 @test21(i32 %a) {
  %shl = shl nsw i32 %a, 2
  %div = sdiv i32 %shl, 12
  ret i32 %div
; CHECK-LABEL: @test21(
; CHECK-NEXT: %div = sdiv i32 %a, 3
; CHECK-NEXT: ret i32 %div
}

define i32 @test22(i32 %a) {
  %mul = mul nsw i32 %a, 3
  %div = sdiv i32 %mul, 12
  ret i32 %div
; CHECK-LABEL: @test22(
; CHECK-NEXT: %div = sdiv i32 %a, 4
; CHECK-NEXT: ret i32 %div
}

define i32 @test23(i32 %a) {
  %shl = shl nuw i32 %a, 2
  %div = udiv i32 %shl, 12
  ret i32 %div
; CHECK-LABEL: @test23(
; CHECK-NEXT: %div = udiv i32 %a, 3
; CHECK-NEXT: ret i32 %div
}

define i32 @test24(i32 %a) {
  %mul = mul nuw i32 %a, 3
  %div = udiv i32 %mul, 12
  ret i32 %div
; CHECK-LABEL: @test24(
; CHECK-NEXT: %div = lshr i32 %a, 2
; CHECK-NEXT: ret i32 %div
}

define i32 @test25(i32 %a) {
  %shl = shl nsw i32 %a, 2
  %div = sdiv i32 %shl, 2
  ret i32 %div
; CHECK-LABEL: @test25(
; CHECK-NEXT: %div = shl nsw i32 %a, 1
; CHECK-NEXT: ret i32 %div
}

define i32 @test26(i32 %a) {
  %mul = mul nsw i32 %a, 12
  %div = sdiv i32 %mul, 3
  ret i32 %div
; CHECK-LABEL: @test26(
; CHECK-NEXT: %div = shl nsw i32 %a, 2
; CHECK-NEXT: ret i32 %div
}

define i32 @test27(i32 %a) {
  %shl = shl nuw i32 %a, 2
  %div = udiv i32 %shl, 2
  ret i32 %div
; CHECK-LABEL: @test27(
; CHECK-NEXT: %div = shl nuw i32 %a, 1
; CHECK-NEXT: ret i32 %div
}

define i32 @test28(i32 %a) {
  %mul = mul nuw i32 %a, 36
  %div = udiv i32 %mul, 3
  ret i32 %div
; CHECK-LABEL: @test28(
; CHECK-NEXT: %div = mul nuw i32 %a, 12
; CHECK-NEXT: ret i32 %div
}

define i32 @test29(i32 %a) {
  %mul = shl nsw i32 %a, 31
  %div = sdiv i32 %mul, -2147483648
  ret i32 %div
; CHECK-LABEL: @test29(
; CHECK-NEXT: %[[and:.*]] = and i32 %a, 1
; CHECK-NEXT: ret i32 %[[and]]
}

define i32 @test30(i32 %a) {
  %mul = shl nuw i32 %a, 31
  %div = udiv i32 %mul, -2147483648
  ret i32 %div
; CHECK-LABEL: @test30(
; CHECK-NEXT: ret i32 %a
}

define <2 x i32> @test31(<2 x i32> %x) {
  %shr = lshr <2 x i32> %x, <i32 31, i32 31>
  %div = udiv <2 x i32> %shr, <i32 2147483647, i32 2147483647>
  ret <2 x i32> %div
; CHECK-LABEL: @test31(
; CHECK-NEXT: ret <2 x i32> zeroinitializer
}

define i32 @test32(i32 %a, i32 %b) {
  %shl = shl i32 2, %b
  %div = lshr i32 %shl, 2
  %div2 = udiv i32 %a, %div
  ret i32 %div2
; CHECK-LABEL: @test32(
; CHECK-NEXT: %[[shl:.*]] = shl i32 2, %b
; CHECK-NEXT: %[[shr:.*]] = lshr i32 %[[shl]], 2
; CHECK-NEXT: %[[div:.*]] = udiv i32 %a, %[[shr]]
; CHECK-NEXT: ret i32
}

define <2 x i64> @test33(<2 x i64> %x) nounwind {
  %shr = lshr exact <2 x i64> %x, <i64 5, i64 5>
  %div = udiv exact <2 x i64> %shr, <i64 6, i64 6>
  ret <2 x i64> %div
; CHECK-LABEL: @test33(
; CHECK-NEXT: udiv exact <2 x i64> %x, <i64 192, i64 192>
; CHECK-NEXT: ret <2 x i64>
}

define <2 x i64> @test34(<2 x i64> %x) nounwind {
  %neg = sub nsw <2 x i64> zeroinitializer, %x
  %div = sdiv exact <2 x i64> %neg, <i64 3, i64 4>
  ret <2 x i64> %div
; CHECK-LABEL: @test34(
; CHECK-NEXT: sdiv exact <2 x i64> %x, <i64 -3, i64 -4>
; CHECK-NEXT: ret <2 x i64>
}

define i32 @test35(i32 %A) {
  %and = and i32 %A, 2147483647
  %mul = sdiv exact i32 %and, 2147483647
  ret i32 %mul
; CHECK-LABEL: @test35(
; CHECK-NEXT: %[[and:.*]]  = and i32 %A, 2147483647
; CHECK-NEXT: %[[udiv:.*]] = udiv exact i32 %[[and]], 2147483647
; CHECK-NEXT: ret i32 %[[udiv]]
}

define i32 @test36(i32 %A) {
  %and = and i32 %A, 2147483647
  %shl = shl nsw i32 1, %A
  %mul = sdiv exact i32 %and, %shl
  ret i32 %mul
; CHECK-LABEL: @test36(
; CHECK-NEXT: %[[and:.*]] = and i32 %A, 2147483647
; CHECK-NEXT: %[[shr:.*]] = lshr exact i32 %[[and]], %A
; CHECK-NEXT: ret i32 %[[shr]]
}

define i32 @test37(i32* %b) {
entry:
  store i32 0, i32* %b, align 4
  %0 = load i32, i32* %b, align 4
  br i1 undef, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %entry
  %mul = mul nsw i32 undef, %0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %t.0 = phi i32 [ %0, %entry ], [ %mul, %lor.rhs ]
  %div = sdiv i32 %t.0, 2
  ret i32 %div
; CHECK-LABEL: @test37(
; CHECK: ret i32 0
}
