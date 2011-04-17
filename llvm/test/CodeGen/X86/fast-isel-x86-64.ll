; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Make sure that fast-isel folds the immediate into the binop even though it
; is non-canonical.
define i32 @test1(i32 %i) nounwind ssp {
  %and = and i32 8, %i
  ret i32 %and
}

; CHECK: test1:
; CHECK: andl	$8, 
