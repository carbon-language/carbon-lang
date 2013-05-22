; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.hoge = type { double, double, double}

define void @zot(%struct.hoge* %arg) {
bb:
  %tmp = load double* undef, align 8
  %tmp1 = fsub double %tmp, undef
  %tmp2 = load double* undef, align 8
  %tmp3 = fsub double %tmp2, undef
  %tmp4 = fmul double %tmp3, undef
  %tmp5 = fmul double %tmp3, undef
  %tmp6 = fsub double %tmp5, undef
  %tmp7 = getelementptr inbounds %struct.hoge* %arg, i64 0, i32 1
  store double %tmp6, double* %tmp7, align 8
  %tmp8 = fmul double %tmp1, undef
  %tmp9 = fsub double %tmp8, undef
  %tmp10 = getelementptr inbounds %struct.hoge* %arg, i64 0, i32 2
  store double %tmp9, double* %tmp10, align 8
  br i1 undef, label %bb11, label %bb12

bb11:                                             ; preds = %bb
  br label %bb14

bb12:                                             ; preds = %bb
  %tmp13 = fmul double undef, %tmp2
  br label %bb14

bb14:                                             ; preds = %bb12, %bb11
  ret void
}
