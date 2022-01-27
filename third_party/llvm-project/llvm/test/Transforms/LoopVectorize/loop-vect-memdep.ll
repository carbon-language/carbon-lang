target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; RUN: opt < %s -S -loop-vectorize -debug-only=loop-vectorize 2>&1 | FileCheck %s
; REQUIRES: asserts
; CHECK: LV: Can't vectorize due to memory conflicts

define void @test_loop_novect(double** %arr, i64 %n) {
for.body.lr.ph:
  %t = load double*, double** %arr, align 8
  br label %for.body

for.body:                                      ; preds = %for.body, %for.body.lr.ph
  %i = phi i64 [ 0, %for.body.lr.ph ], [ %i.next, %for.body ]
  %a = getelementptr inbounds double, double* %t, i64 %i
  %i.next = add nuw nsw i64 %i, 1
  %a.next = getelementptr inbounds double, double* %t, i64 %i.next
  %t1 = load double, double* %a, align 8
  %t2 = load double, double* %a.next, align 8
  store double %t1, double* %a.next, align 8
  store double %t2, double* %a, align 8
  %c = icmp eq i64 %i, %n
  br i1 %c, label %final, label %for.body

final:                                   ; preds = %for.body
  ret void
}
