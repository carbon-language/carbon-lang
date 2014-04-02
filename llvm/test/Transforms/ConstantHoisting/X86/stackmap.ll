; RUN: opt -S -consthoist < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Test if the 3rd argument of a stackmap is hoisted.
define i128 @test1(i128 %a) {
; CHECK-LABEL:  @test1
; CHECK:        %const = bitcast i128 13464618275673403322 to i128
; CHECK:        tail call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 1, i32 24, i128 %const)
entry:
  %0 = add i128 %a, 13464618275673403322
  tail call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 1, i32 24, i128 13464618275673403322)
  ret i128 %0
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
