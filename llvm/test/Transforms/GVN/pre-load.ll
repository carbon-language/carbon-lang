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
  store i32 0, i32* %p
  br label %block4

block4:
  %PRE = load i32* %p
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32
; CHECK-NEXT: ret i32
}

define i32 @test2(i32* %p, i32* %q, i1 %C) {
; CHECK: @test2
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32* %q

block3:
  store i32 0, i32* %p
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %PRE = load i32* %P2
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}

define i32 @test3(i32* %p, i32* %q, i32** %Hack, i1 %C) {
; CHECK: @test3
block1:
  %B = getelementptr i32* %q, i32 1
  store i32* %B, i32** %Hack
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32* %B

block3:
  %A = getelementptr i32* %p, i32 1
  store i32 0, i32* %A
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %P3 = getelementptr i32* %P2, i32 1
  %PRE = load i32* %P3
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}
