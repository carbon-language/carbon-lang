; RUN: opt -basicaa -load-combine -instcombine -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @test1(i32* nocapture readonly noalias %a, i32* nocapture readonly noalias %b) {
; CHECK-LABEL: @test1

; CHECK: load i64*
; CHECK: ret i64

  %load1 = load i32* %a, align 4
  %conv = zext i32 %load1 to i64
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %load1, i32* %b, align 4
  %load2 = load i32* %arrayidx1, align 4
  %conv2 = zext i32 %load2 to i64
  %shl = shl nuw i64 %conv2, 32
  %add = or i64 %shl, %conv
  ret i64 %add
}

define i64 @test2(i32* nocapture readonly %a, i32* nocapture readonly %b) {
; CHECK-LABEL: @test2

; CHECK: load i32*
; CHECK: load i32*
; CHECK: ret i64

  %load1 = load i32* %a, align 4
  %conv = zext i32 %load1 to i64
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %load1, i32* %b, align 4
  %load2 = load i32* %arrayidx1, align 4
  %conv2 = zext i32 %load2 to i64
  %shl = shl nuw i64 %conv2, 32
  %add = or i64 %shl, %conv
  ret i64 %add
}

