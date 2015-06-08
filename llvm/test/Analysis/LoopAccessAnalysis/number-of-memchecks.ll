; RUN: opt -loop-accesses -analyze < %s | FileCheck %s

; 3 reads and 3 writes should need 12 memchecks

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnueabi"

; CHECK: Memory dependences are safe with run-time checks
; Memory dependecies have labels starting from 0, so in
; order to verify that we have n checks, we look for
; (n-1): and not n:.

; CHECK: Run-time memory checks:
; CHECK-NEXT: 0:
; CHECK: 11:
; CHECK-NOT: 12:

define void @testf(i16* %a,
               i16* %b,
               i16* %c,
               i16* %d,
               i16* %e,
               i16* %f) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %add = add nuw nsw i64 %ind, 1

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %ind
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %ind
  %loadC = load i16, i16* %arrayidxC, align 2

  %mul = mul i16 %loadB, %loadA
  %mul1 = mul i16 %mul, %loadC

  %arrayidxD = getelementptr inbounds i16, i16* %d, i64 %ind
  store i16 %mul1, i16* %arrayidxD, align 2

  %arrayidxE = getelementptr inbounds i16, i16* %e, i64 %ind
  store i16 %mul, i16* %arrayidxE, align 2

  %arrayidxF = getelementptr inbounds i16, i16* %f, i64 %ind
  store i16 %mul1, i16* %arrayidxF, align 2

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
