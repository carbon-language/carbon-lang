; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @main() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.end44, %entry
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %if.then25, %for.body
  br label %for.body6

for.body6:                                        ; preds = %for.inc21, %for.cond4.preheader
  br label %for.body12

for.body12:                                       ; preds = %if.end, %for.body6
  %fZImg.069 = phi double [ undef, %for.body6 ], [ %add19, %if.end ]
  %fZReal.068 = phi double [ undef, %for.body6 ], [ %add20, %if.end ]
  %mul13 = fmul double %fZReal.068, %fZReal.068
  %mul14 = fmul double %fZImg.069, %fZImg.069
  %add15 = fadd double %mul13, %mul14
  %cmp16 = fcmp ogt double %add15, 4.000000e+00
  br i1 %cmp16, label %for.inc21, label %if.end

if.end:                                           ; preds = %for.body12
  %mul18 = fmul double undef, %fZImg.069
  %add19 = fadd double undef, %mul18
  %sub = fsub double %mul13, %mul14
  %add20 = fadd double undef, %sub
  br i1 undef, label %for.body12, label %for.inc21

for.inc21:                                        ; preds = %if.end, %for.body12
  br i1 undef, label %for.end23, label %for.body6

for.end23:                                        ; preds = %for.inc21
  br i1 undef, label %if.then25, label %if.then26

if.then25:                                        ; preds = %for.end23
  br i1 undef, label %for.end44, label %for.cond4.preheader

if.then26:                                        ; preds = %for.end23
  unreachable

for.end44:                                        ; preds = %if.then25
  br i1 undef, label %for.end48, label %for.body

for.end48:                                        ; preds = %for.end44
  ret void
}

%struct.hoge = type { double, double, double}

define void @zot(%struct.hoge* %arg) {
bb:
  %tmp = load double, double* undef, align 8
  %tmp1 = fsub double %tmp, undef
  %tmp2 = load double, double* undef, align 8
  %tmp3 = fsub double %tmp2, undef
  %tmp4 = fmul double %tmp3, undef
  %tmp5 = fmul double %tmp3, undef
  %tmp6 = fsub double %tmp5, undef
  %tmp7 = getelementptr inbounds %struct.hoge, %struct.hoge* %arg, i64 0, i32 1
  store double %tmp6, double* %tmp7, align 8
  %tmp8 = fmul double %tmp1, undef
  %tmp9 = fsub double %tmp8, undef
  %tmp10 = getelementptr inbounds %struct.hoge, %struct.hoge* %arg, i64 0, i32 2
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


%struct.rc4_state.0.24 = type { i32, i32, [256 x i32] }

define void @rc4_crypt(%struct.rc4_state.0.24* nocapture %s) {
entry:
  %x1 = getelementptr inbounds %struct.rc4_state.0.24, %struct.rc4_state.0.24* %s, i64 0, i32 0
  %y2 = getelementptr inbounds %struct.rc4_state.0.24, %struct.rc4_state.0.24* %s, i64 0, i32 1
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

