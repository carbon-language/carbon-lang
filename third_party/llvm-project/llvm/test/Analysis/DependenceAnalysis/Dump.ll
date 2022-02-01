; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

;; Test to make sure the dump shows the src and dst
;; instructions (including call instructions).
;;
;; void bar(float * restrict A);
;; void foo(float * restrict A, int n) {
;;   for (int i = 0; i < n; i++) {
;;     A[i] = i;
;;     bar(A);
;;   }
;; }

; CHECK-LABEL: foo

; CHECK: Src:  store float %conv, float* %arrayidx, align 4 --> Dst:  store float %conv, float* %arrayidx, align 4
; CHECK-NEXT:   da analyze - none!
; CHECK-NEXT: Src:  store float %conv, float* %arrayidx, align 4 --> Dst:  call void @bar(float* %A)
; CHECK-NEXT:   da analyze - confused!
; CHECK-NEXT: Src:  call void @bar(float* %A) --> Dst:  call void @bar(float* %A)
; CHECK-NEXT:   da analyze - confused!

define void @foo(float* noalias %A, i32 signext %n) {
entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %conv = sitofp i32 %i.02 to float
  %idxprom = zext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds float, float* %A, i64 %idxprom
  store float %conv, float* %arrayidx, align 4
  call void @bar(float* %A) #3
  %inc = add nuw nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

declare void @bar(float*)
