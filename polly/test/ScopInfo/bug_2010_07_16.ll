; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir -analyze %s | FileCheck %s

; ModuleID = '/home/ether/unexpected_parameter.ll'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @mdct_sub48([2 x [576 x double]]* %mdct_freq) nounwind {
entry:
  br label %bb54

bb4:                                              ; preds = %bb54, %bb49
  br label %bb6

bb6:                                              ; preds = %bb6, %bb4
  br i1 undef, label %bb6, label %bb48

bb24:                                             ; preds = %bb48
  br i1 false, label %bb47, label %bb46

bb40:                                             ; preds = %bb46
  %0 = load double* %scevgep74, align 8           ; <double> [#uses=0]
  %indvar.next62 = add i64 %indvar61, 1           ; <i64> [#uses=1]
  br label %bb46

bb46:                                             ; preds = %bb40, %bb24
  %indvar61 = phi i64 [ %indvar.next62, %bb40 ], [ 0, %bb24 ] ; <i64> [#uses=1]
  %scevgep74 = getelementptr [2 x [576 x double]]* %mdct_freq, i64 0, i64 %indvar1, i64 0 ; <double*> [#uses=1]
  store double undef, double* %scevgep74, align 8
  br i1 false, label %bb40, label %bb47

bb47:                                             ; preds = %bb46, %bb24
  br label %bb48

bb48:                                             ; preds = %bb47, %bb6
  br i1 false, label %bb24, label %bb49

bb49:                                             ; preds = %bb48
  br i1 undef, label %bb4, label %bb53

bb53:                                             ; preds = %bb49
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %bb54

bb54:                                             ; preds = %bb53, %entry
  %indvar1 = phi i64 [ %indvar.next2, %bb53 ], [ 0, %entry ] ; <i64> [#uses=2]
  br i1 undef, label %bb4, label %return

return:                                           ; preds = %bb54
  ret void
}

; CHECK: Scop: bb24 => bb48.region      Parameters: ({0,+,1}<%bb54>, ), Max Loop Depth: 1
