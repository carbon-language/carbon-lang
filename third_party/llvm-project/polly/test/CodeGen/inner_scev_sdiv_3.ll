; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; This test case has a inner SCEV sdiv that will escape the SCoP. Just check we
; do not crash and generate valid code.
;
; CHECK:    polly.split_new_and_old:
;
target triple = "x86_64-unknown-linux-gnu"

define i64 @_vorbis_apply_window(float* %d, i64 %param) {
entry:
  %0 = load float*, float** undef, align 8
  %div23.neg = sdiv i64 0, -4
  %sub24 = add i64 0, %div23.neg
  br label %for.cond.30.preheader

for.cond.30.preheader:                            ; preds = %for.body, %entry
  %sext = shl i64 %sub24, 32
  %conv48.74 = ashr exact i64 %sext, 32
  %div43 = sdiv i64 %param, 2
  %cmp49.75 = icmp slt i64 %conv48.74, 0
  br i1 %cmp49.75, label %for.body.51.lr.ph, label %for.cond.60.preheader

for.body.51.lr.ph:                                ; preds = %for.cond.30.preheader
  %div44 = sdiv i64 %div43, 2
  %sub45 = add nsw i64 %div44, 4294967295
  %1 = trunc i64 %sub45 to i32
  %2 = sext i32 %1 to i64
  br label %for.body.51

for.cond.60.preheader:                            ; preds = %for.body.51, %for.cond.30.preheader
  %div44.m = phi i64 [%div44, %for.body.51], [ 0, %for.cond.30.preheader]
  br i1 true, label %end, label %for.cond.30.preheader

end:
  ret i64 %div44.m

for.body.51:                                      ; preds = %for.body.51, %for.body.51.lr.ph
  %indvars.iv86 = phi i64 [ %2, %for.body.51.lr.ph ], [ undef, %for.body.51 ]
  %arrayidx53 = getelementptr inbounds float, float* %0, i64 %indvars.iv86
  %3 = load float, float* %arrayidx53, align 4
  %arrayidx55 = getelementptr inbounds float, float* %d, i64 0
  %mul56 = fmul float %3, undef
  store float %mul56, float* %arrayidx55, align 4
  br i1 false, label %for.body.51, label %for.cond.60.preheader
}
