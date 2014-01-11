; RUN: opt < %s -loop-vectorize -force-vector-unroll=1 -force-vector-width=2 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure that we can handle multiple integer induction variables.
; CHECK-LABEL: @multi_int_induction(
; CHECK: vector.body:
; CHECK:  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:  %normalized.idx = sub i64 %index, 0
; CHECK:  %[[VAR:.*]] = trunc i64 %normalized.idx to i32
; CHECK:  %offset.idx = add i32 190, %[[VAR]]
define void @multi_int_induction(i32* %A, i32 %N) {
for.body.lr.ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %count.09 = phi i32 [ 190, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 %count.09, i32* %arrayidx2, align 4
  %inc = add nsw i32 %count.09, 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; RUN: opt < %s -loop-vectorize -force-vector-unroll=1 -force-vector-width=2 -instcombine -S | FileCheck %s --check-prefix=IND

; Make sure we remove unneeded vectorization of induction variables.
; In order for instcombine to cleanup the vectorized induction variables that we
; create in the loop vectorizer we need to perform some form of redundancy
; elimination to get rid of multiple uses.

; IND-LABEL: scalar_use

; IND:     br label %vector.body
; IND:     vector.body:
;   Vectorized induction variable.
; IND-NOT:  insertelement <2 x i64>
; IND-NOT:  shufflevector <2 x i64>
; IND:     br {{.*}}, label %vector.body

define void @scalar_use(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float* %a, i64 %ind.sum
  %l1 = load float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float* %a, i64 %ind.sum2
  %l2 = load float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}


; Make sure that the loop exit count computation does not overflow for i8 and
; i16. The exit count of these loops is i8/i16 max + 1. If we don't cast the
; induction variable to a bigger type the exit count computation will overflow
; to 0.
; PR17532

; CHECK-LABEL: i8_loop
; CHECK: icmp eq i32 {{.*}}, 256
define i32 @i8_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i8 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i8 %b.0, -1
  %4 = icmp eq i8 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; CHECK-LABEL: i16_loop
; CHECK: icmp eq i32 {{.*}}, 65536

define i32 @i16_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i16 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i16 %b.0, -1
  %4 = icmp eq i16 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}
