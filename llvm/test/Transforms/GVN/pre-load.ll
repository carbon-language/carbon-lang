; RUN: opt < %s -gvn -enable-load-pre -S | FileCheck %s

define i32 @test1(i32* %p, i1 %C) {
; CHECK: @test1
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32* %p

block3:
  %b = bitcast i32 0 to i32
  store i32 %b, i32* %p
  br label %block4

block4:
  %PRE = load i32* %p
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32
; CHECK-NEXT: ret i32
}
