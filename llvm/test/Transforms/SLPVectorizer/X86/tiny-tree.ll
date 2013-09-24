target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"
; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=-5 -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 | FileCheck %s

%t1 = type {%t2, double, double, i8 }
%t2 = type { double, double, double }
%t3 = type { double, double, i8 }

; We check that SLP vectorizer will not try to vectorize tiny trees
; even with a negative threshold.  
; CHECK: tiny_tree_test
; CHECK-NOT:  <2 x double>
; CHECK: ret 

define void @tiny_tree_test(%t3* %this, %t1* %m)  align 2 {
entry:
  %m41.i = getelementptr inbounds %t1* %m, i64 0, i32 0, i32 1
  %0 = load double* %m41.i, align 8
  %_tx = getelementptr inbounds %t3* %this, i64 0, i32 0
  store double %0, double* %_tx, align 8
  %m42.i = getelementptr inbounds %t1* %m, i64 0, i32 0, i32 2
  %1 = load double* %m42.i, align 8
  %_ty = getelementptr inbounds %t3* %this, i64 0, i32 1
  store double %1, double* %_ty, align 8
  ret void
}

