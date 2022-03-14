; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"


@flag = external global i1


define i32 @test_sextloadi1_32() {
entry:
  %0 = load i1, i1* @flag, align 4
; CHECK: memub
  %1 = sext i1 %0 to i32
  ret i32 %1
}



define i16 @test_zextloadi1_16() {
entry:
  %0 = load i1, i1* @flag, align 4
; CHECK: memub
  %1 = zext i1 %0 to i16
  ret i16 %1
}


define i32 @test_zextloadi1_32() {
entry:
  %0 = load i1, i1* @flag, align 4
; CHECK: memub
  %1 = zext i1 %0 to i32
  ret i32 %1
}


define i64 @test_zextloadi1_64() {
entry:
  %0 = load i1, i1* @flag, align 4
; CHECK: memub
  %1 = zext i1 %0 to i64
  ret i64 %1
}


