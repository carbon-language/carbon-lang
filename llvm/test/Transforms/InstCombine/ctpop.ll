; RUN: opt < %s -S -instcombine | FileCheck %s

declare i32 @llvm.ctpop.i32(i32)
declare i8 @llvm.ctpop.i8(i8)
declare void @llvm.assume(i1)

define i1 @test1(i32 %arg) {
; CHECK: @test1
; CHECK: ret i1 false
  %and = and i32 %arg, 15
  %cnt = call i32 @llvm.ctpop.i32(i32 %and)
  %res = icmp eq i32 %cnt, 9
  ret i1 %res
}

define i1 @test2(i32 %arg) {
; CHECK: @test2
; CHECK: ret i1 false
  %and = and i32 %arg, 1
  %cnt = call i32 @llvm.ctpop.i32(i32 %and)
  %res = icmp eq i32 %cnt, 2
  ret i1 %res
}

define i1 @test3(i32 %arg) {
; CHECK: @test3
; CHECK: ret i1 false
  ;; Use an assume to make all the bits known without triggering constant 
  ;; folding.  This is trying to hit a corner case where we have to avoid
  ;; taking the log of 0.
  %assume = icmp eq i32 %arg, 0
  call void @llvm.assume(i1 %assume)
  %cnt = call i32 @llvm.ctpop.i32(i32 %arg)
  %res = icmp eq i32 %cnt, 2
  ret i1 %res
}

; Negative test for when we know nothing
define i1 @test4(i8 %arg) {
; CHECK: @test4
; CHECK: ret i1 %res
  %cnt = call i8 @llvm.ctpop.i8(i8 %arg)
  %res = icmp eq i8 %cnt, 2
  ret i1 %res
}
