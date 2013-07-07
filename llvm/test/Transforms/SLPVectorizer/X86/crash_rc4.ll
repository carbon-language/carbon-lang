; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.rc4_state.0.24 = type { i32, i32, [256 x i32] }

define void @rc4_crypt(%struct.rc4_state.0.24* nocapture %s) {
entry:
  %x1 = getelementptr inbounds %struct.rc4_state.0.24* %s, i64 0, i32 0
  %y2 = getelementptr inbounds %struct.rc4_state.0.24* %s, i64 0, i32 1
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %x.045 = phi i32 [ %conv4, %for.body ], [ undef, %entry ]
  %conv4 = and i32 undef, 255
  %conv7 = and i32 undef, 255
  %idxprom842 = zext i32 %conv7 to i64
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %x.0.lcssa = phi i32 [ undef, %entry ], [ %conv4, %for.body ]
  %y.0.lcssa = phi i32 [ undef, %entry ], [ %conv7, %for.body ]
  store i32 %x.0.lcssa, i32* %x1, align 4
  store i32 %y.0.lcssa, i32* %y2, align 4
  ret void
}

