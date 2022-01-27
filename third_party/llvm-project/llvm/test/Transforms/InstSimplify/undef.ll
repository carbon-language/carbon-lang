; RUN: opt -instsimplify -S < %s | FileCheck %s

define i64 @test0() {
; CHECK-LABEL: @test0(
; CHECK:         ret i64 undef
;
  %r = mul i64 undef, undef
  ret i64 %r
}

define i64 @test1() {
; CHECK-LABEL: @test1(
; CHECK:         ret i64 undef
;
  %r = mul i64 3, undef
  ret i64 %r
}

define i64 @test2() {
; CHECK-LABEL: @test2(
; CHECK:         ret i64 undef
;
  %r = mul i64 undef, 3
  ret i64 %r
}

define i64 @test3() {
; CHECK-LABEL: @test3(
; CHECK:         ret i64 0
;
  %r = mul i64 undef, 6
  ret i64 %r
}

define i64 @test4() {
; CHECK-LABEL: @test4(
; CHECK:         ret i64 0
;
  %r = mul i64 6, undef
  ret i64 %r
}

define i64 @test5() {
; CHECK-LABEL: @test5(
; CHECK:         ret i64 undef
;
  %r = and i64 undef, undef
  ret i64 %r
}

define i64 @test6() {
; CHECK-LABEL: @test6(
; CHECK:         ret i64 undef
;
  %r = or i64 undef, undef
  ret i64 %r
}

define i64 @test7() {
; CHECK-LABEL: @test7(
; CHECK:         ret i64 undef
;
  %r = udiv i64 undef, 1
  ret i64 %r
}

define i64 @test8() {
; CHECK-LABEL: @test8(
; CHECK:         ret i64 undef
;
  %r = sdiv i64 undef, 1
  ret i64 %r
}

define i64 @test9() {
; CHECK-LABEL: @test9(
; CHECK:         ret i64 0
;
  %r = urem i64 undef, 1
  ret i64 %r
}

define i64 @test10() {
; CHECK-LABEL: @test10(
; CHECK:         ret i64 0
;
  %r = srem i64 undef, 1
  ret i64 %r
}

define i64 @test11() {
; CHECK-LABEL: @test11(
; CHECK:         ret i64 poison
;
  %r = shl i64 undef, undef
  ret i64 %r
}

define i64 @test11b(i64 %a) {
; CHECK-LABEL: @test11b(
; CHECK:         ret i64 poison
;
  %r = shl i64 %a, undef
  ret i64 %r
}

define i64 @test12() {
; CHECK-LABEL: @test12(
; CHECK:         ret i64 poison
;
  %r = ashr i64 undef, undef
  ret i64 %r
}

define i64 @test12b(i64 %a) {
; CHECK-LABEL: @test12b(
; CHECK:         ret i64 poison
;
  %r = ashr i64 %a, undef
  ret i64 %r
}

define i64 @test13() {
; CHECK-LABEL: @test13(
; CHECK:         ret i64 poison
;
  %r = lshr i64 undef, undef
  ret i64 %r
}

define i64 @test13b(i64 %a) {
; CHECK-LABEL: @test13b(
; CHECK:         ret i64 poison
;
  %r = lshr i64 %a, undef
  ret i64 %r
}

define i1 @test14() {
; CHECK-LABEL: @test14(
; CHECK:         ret i1 undef
;
  %r = icmp slt i64 undef, undef
  ret i1 %r
}

define i1 @test15() {
; CHECK-LABEL: @test15(
; CHECK:         ret i1 undef
;
  %r = icmp ult i64 undef, undef
  ret i1 %r
}

define i64 @test16(i64 %a) {
; CHECK-LABEL: @test16(
; CHECK:         ret i64 undef
;
  %r = select i1 undef, i64 %a, i64 undef
  ret i64 %r
}

define i64 @test17(i64 %a) {
; CHECK-LABEL: @test17(
; CHECK:         ret i64 undef
;
  %r = select i1 undef, i64 undef, i64 %a
  ret i64 %r
}

define i64 @test18(i64 %a) {
; CHECK-LABEL: @test18(
; CHECK:         [[R:%.*]] = call i64 undef(i64 %a)
; CHECK-NEXT:    ret i64 poison
;
  %r = call i64 (i64) undef(i64 %a)
  ret i64 %r
}

define <4 x i8> @test19(<4 x i8> %a) {
; CHECK-LABEL: @test19(
; CHECK:         ret <4 x i8> poison
;
  %b = shl <4 x i8> %a, <i8 8, i8 9, i8 undef, i8 -1>
  ret <4 x i8> %b
}

define i32 @test20(i32 %a) {
; CHECK-LABEL: @test20(
; CHECK:         ret i32 poison
;
  %b = udiv i32 %a, 0
  ret i32 %b
}

define <2 x i32> @test20vec(<2 x i32> %a) {
; CHECK-LABEL: @test20vec(
; CHECK-NEXT:    ret <2 x i32> poison
;
  %b = udiv <2 x i32> %a, zeroinitializer
  ret <2 x i32> %b
}

define i32 @test21(i32 %a) {
; CHECK-LABEL: @test21(
; CHECK:         ret i32 poison
;
  %b = sdiv i32 %a, 0
  ret i32 %b
}

define <2 x i32> @test21vec(<2 x i32> %a) {
; CHECK-LABEL: @test21vec(
; CHECK-NEXT:    ret <2 x i32> poison
;
  %b = sdiv <2 x i32> %a, zeroinitializer
  ret <2 x i32> %b
}

define i32 @test22(i32 %a) {
; CHECK-LABEL: @test22(
; CHECK:         ret i32 undef
;
  %b = ashr exact i32 undef, %a
  ret i32 %b
}

define i32 @test23(i32 %a) {
; CHECK-LABEL: @test23(
; CHECK:         ret i32 undef
;
  %b = lshr exact i32 undef, %a
  ret i32 %b
}

define i32 @test24() {
; CHECK-LABEL: @test24(
; CHECK:         ret i32 poison
;
  %b = udiv i32 undef, 0
  ret i32 %b
}

define i32 @test25() {
; CHECK-LABEL: @test25(
; CHECK:         ret i32 poison
;
  %b = lshr i32 0, undef
  ret i32 %b
}

define i32 @test26() {
; CHECK-LABEL: @test26(
; CHECK:         ret i32 poison
;
  %b = ashr i32 0, undef
  ret i32 %b
}

define i32 @test27() {
; CHECK-LABEL: @test27(
; CHECK:         ret i32 poison
;
  %b = shl i32 0, undef
  ret i32 %b
}

define i32 @test28(i32 %a) {
; CHECK-LABEL: @test28(
; CHECK:         ret i32 undef
;
  %b = shl nsw i32 undef, %a
  ret i32 %b
}

define i32 @test29(i32 %a) {
; CHECK-LABEL: @test29(
; CHECK:         ret i32 undef
;
  %b = shl nuw i32 undef, %a
  ret i32 %b
}

define i32 @test30(i32 %a) {
; CHECK-LABEL: @test30(
; CHECK:         ret i32 undef
;
  %b = shl nsw nuw i32 undef, %a
  ret i32 %b
}

define i32 @test31(i32 %a) {
; CHECK-LABEL: @test31(
; CHECK:         ret i32 0
;
  %b = shl i32 undef, %a
  ret i32 %b
}

define i32 @test32(i32 %a) {
; CHECK-LABEL: @test32(
; CHECK:         ret i32 undef
;
  %b = shl i32 undef, 0
  ret i32 %b
}

define i32 @test33(i32 %a) {
; CHECK-LABEL: @test33(
; CHECK:         ret i32 undef
;
  %b = ashr i32 undef, 0
  ret i32 %b
}

define i32 @test34(i32 %a) {
; CHECK-LABEL: @test34(
; CHECK:         ret i32 undef
;
  %b = lshr i32 undef, 0
  ret i32 %b
}

define i32 @test35(<4 x i32> %V) {
; CHECK-LABEL: @test35(
; CHECK:         ret i32 poison
;
  %b = extractelement <4 x i32> %V, i32 4
  ret i32 %b
}

define i32 @test36(i32 %V) {
; CHECK-LABEL: @test36(
; CHECK:         ret i32 undef
;
  %b = extractelement <4 x i32> undef, i32 %V
  ret i32 %b
}

define i32 @test37() {
; CHECK-LABEL: @test37(
; CHECK:         ret i32 poison
;
  %b = udiv i32 undef, undef
  ret i32 %b
}

define i32 @test38(i32 %a) {
; CHECK-LABEL: @test38(
; CHECK:         ret i32 poison
;
  %b = udiv i32 %a, undef
  ret i32 %b
}

define i32 @test39() {
; CHECK-LABEL: @test39(
; CHECK:         ret i32 poison
;
  %b = udiv i32 0, undef
  ret i32 %b
}
