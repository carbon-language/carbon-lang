; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test1
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test2
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

define i32 @test3(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test3
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test4(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test4
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

; PR4837
define <2 x i1> @test5(<2 x i64> %x) {
entry:
  %V = icmp eq <2 x i64> %x, undef
  ret <2 x i1> %V
; CHECK: @test5
; CHECK: ret <2 x i1> <i1 true, i1 true>
}

define i32 @test6(i32 %a, i32 %b) {
        %c = icmp sle i32 %a, -1
        %d = zext i1 %c to i32
        %e = sub i32 0, %d
        %f = and i32 %e, %b
        ret i32 %f
; CHECK: @test6
; CHECK-NEXT: ashr i32 %a, 31
; CHECK-NEXT: %f = and i32 %e, %b
; CHECK-NEXT: ret i32 %f
}


define i1 @test7(i32 %x) {
entry:
  %a = add i32 %x, -1
  %b = icmp ult i32 %a, %x
  ret i1 %b
; CHECK: @test7
; CHECK: %b = icmp ne i32 %x, 0
; CHECK: ret i1 %b
}

define i1 @test8(i32 %x){
entry:
  %a = add i32 %x, -1 
  %b = icmp eq i32 %a, %x
  ret i1 %b
; CHECK: @test8
; CHECK: ret i1 false
}

define i1 @test9(i32 %x)  {
entry:
  %a = add i32 %x, -2
  %b = icmp ugt i32 %x, %a 
  ret i1 %b
; CHECK: @test9
; CHECK: icmp ugt i32 %x, 1
; CHECK: ret i1 %b
}

define i1 @test10(i32 %x){
entry:
  %a = add i32 %x, -1      
  %b = icmp slt i32 %a, %x 
  ret i1 %b
  
; CHECK: @test10
; CHECK: %b = icmp ne i32 %x, -2147483648
; CHECK: ret i1 %b
}

define i1 @test11(i32 %x) {
  %a = add nsw i32 %x, 8
  %b = icmp slt i32 %x, %a
  ret i1 %b
; CHECK: @test11  
; CHECK: ret i1 true
}

; PR6195
define i1 @test12(i1 %A) {
  %S = select i1 %A, i64 -4294967295, i64 8589934591
  %B = icmp ne i64 bitcast (<2 x i32> <i32 1, i32 -1> to i64), %S
  ret i1 %B
; CHECK: @test12
; CHECK-NEXT: %B = select i1
; CHECK-NEXT: ret i1 %B
}

; PR6481
define i1 @test13(i8 %X) nounwind readnone {
entry:
        %cmp = icmp slt i8 undef, %X
        ret i1 %cmp
; CHECK: @test13
; CHECK: ret i1 false
}

define i1 @test14(i8 %X) nounwind readnone {
entry:
        %cmp = icmp slt i8 undef, -128
        ret i1 %cmp
; CHECK: @test14
; CHECK: ret i1 false
}

define i1 @test15() nounwind readnone {
entry:
        %cmp = icmp eq i8 undef, -128
        ret i1 %cmp
; CHECK: @test15
; CHECK: ret i1 undef
}

define i1 @test16() nounwind readnone {
entry:
        %cmp = icmp ne i8 undef, -128
        ret i1 %cmp
; CHECK: @test16
; CHECK: ret i1 undef
}

define i1 @test17(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
; CHECK: @test17
; CHECK-NEXT: %cmp = icmp ne i32 %x, 3
}


define i1 @test18(i32 %x) nounwind {
  %sh = lshr i32 8, %x
  %and = and i32 %sh, 1
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
; CHECK: @test18
; CHECK-NEXT: %cmp = icmp ne i32 %x, 3
}

define i1 @test19(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp eq i32 %and, 8
  ret i1 %cmp
; CHECK: @test19
; CHECK-NEXT: %cmp = icmp eq i32 %x, 3
}

define i1 @test20(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
; CHECK: @test20
; CHECK-NEXT: %cmp = icmp eq i32 %x, 3
}

define i1 @test21(i8 %x, i8 %y) {
; CHECK: @test21
; CHECK-NOT: or i8
; CHECK: icmp ugt
  %A = or i8 %x, 1
  %B = icmp ugt i8 %A, 3
  ret i1 %B
}

define i1 @test22(i8 %x, i8 %y) {
; CHECK: @test22
; CHECK-NOT: or i8
; CHECK: icmp ult
  %A = or i8 %x, 1
  %B = icmp ult i8 %A, 4
  ret i1 %B
}

; PR2740
; CHECK: @test23
; CHECK: icmp sgt i32 %x, 1328634634
define i1 @test23(i32 %x) nounwind {
	%i3 = sdiv i32 %x, -1328634635
	%i4 = icmp eq i32 %i3, -1
	ret i1 %i4
}

@X = global [1000 x i32] zeroinitializer

; PR8882
; CHECK: @test24
; CHECK:    %cmp = icmp eq i64 %i, 1000
; CHECK:   ret i1 %cmp
define i1 @test24(i64 %i) {
  %p1 = getelementptr inbounds i32* getelementptr inbounds ([1000 x i32]* @X, i64 0, i64 0), i64 %i
  %cmp = icmp eq i32* %p1, getelementptr inbounds ([1000 x i32]* @X, i64 1, i64 0)
  ret i1 %cmp
}

; CHECK: @test25
; X + Z > Y + Z -> X > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test25(i32 %x, i32 %y, i32 %z) {
  %lhs = add nsw i32 %x, %z
  %rhs = add nsw i32 %y, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK: @test26
; X + Z > Y + Z -> X > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test26(i32 %x, i32 %y, i32 %z) {
  %lhs = add nuw i32 %x, %z
  %rhs = add nuw i32 %y, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK: @test27
; X - Z > Y - Z -> X > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test27(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nsw i32 %x, %z
  %rhs = sub nsw i32 %y, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK: @test28
; X - Z > Y - Z -> X > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test28(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nuw i32 %x, %z
  %rhs = sub nuw i32 %y, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK: @test29
; X + Y > X -> Y > 0 if there is no overflow.
; CHECK: %c = icmp sgt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test29(i32 %x, i32 %y) {
  %lhs = add nsw i32 %x, %y
  %c = icmp sgt i32 %lhs, %x
  ret i1 %c
}

; CHECK: @test30
; X + Y > X -> Y > 0 if there is no overflow.
; CHECK: %c = icmp ne i32 %y, 0
; CHECK: ret i1 %c
define i1 @test30(i32 %x, i32 %y) {
  %lhs = add nuw i32 %x, %y
  %c = icmp ugt i32 %lhs, %x
  ret i1 %c
}

; CHECK: @test31
; X > X + Y -> 0 > Y if there is no overflow.
; CHECK: %c = icmp slt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test31(i32 %x, i32 %y) {
  %rhs = add nsw i32 %x, %y
  %c = icmp sgt i32 %x, %rhs
  ret i1 %c
}

; CHECK: @test32
; X > X + Y -> 0 > Y if there is no overflow.
; CHECK: ret i1 false
define i1 @test32(i32 %x, i32 %y) {
  %rhs = add nuw i32 %x, %y
  %c = icmp ugt i32 %x, %rhs
  ret i1 %c
}

; CHECK: @test33
; X - Y > X -> 0 > Y if there is no overflow.
; CHECK: %c = icmp slt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test33(i32 %x, i32 %y) {
  %lhs = sub nsw i32 %x, %y
  %c = icmp sgt i32 %lhs, %x
  ret i1 %c
}

; CHECK: @test34
; X - Y > X -> 0 > Y if there is no overflow.
; CHECK: ret i1 false
define i1 @test34(i32 %x, i32 %y) {
  %lhs = sub nuw i32 %x, %y
  %c = icmp ugt i32 %lhs, %x
  ret i1 %c
}

; CHECK: @test35
; X > X - Y -> Y > 0 if there is no overflow.
; CHECK: %c = icmp sgt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test35(i32 %x, i32 %y) {
  %rhs = sub nsw i32 %x, %y
  %c = icmp sgt i32 %x, %rhs
  ret i1 %c
}

; CHECK: @test36
; X > X - Y -> Y > 0 if there is no overflow.
; CHECK: %c = icmp ne i32 %y, 0
; CHECK: ret i1 %c
define i1 @test36(i32 %x, i32 %y) {
  %rhs = sub nuw i32 %x, %y
  %c = icmp ugt i32 %x, %rhs
  ret i1 %c
}

; CHECK: @test37
; X - Y > X - Z -> Z > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %z, %y
; CHECK: ret i1 %c
define i1 @test37(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nsw i32 %x, %y
  %rhs = sub nsw i32 %x, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK: @test38
; X - Y > X - Z -> Z > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %z, %y
; CHECK: ret i1 %c
define i1 @test38(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nuw i32 %x, %y
  %rhs = sub nuw i32 %x, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}
