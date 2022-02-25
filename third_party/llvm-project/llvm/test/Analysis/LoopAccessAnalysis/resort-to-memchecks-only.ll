; RUN: opt -loop-accesses -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -passes='require<scalar-evolution>,require<aa>,loop(print-access-info)' -disable-output  < %s 2>&1 | FileCheck %s

; We give up analyzing the dependences in this loop due to non-constant
; distance between A[i+offset] and A[i] and add memchecks to prove
; independence.  Make sure that no dependences are reported in
; this case.
;
;   for (i = 0; i < n; i++)
;    A[i + offset] = A[i] * B[i] * C[i];

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK: Memory dependences are safe with run-time checks
; CHECK-NEXT: Dependences:
; CHECK-NEXT: Run-time memory checks:
; CHECK-NEXT: 0:
; CHECK-NEXT: Comparing group
; CHECK-NEXT:   %arrayidxA2 = getelementptr inbounds i16, i16* %a, i64 %idx
; CHECK-NEXT: Against group
; CHECK-NEXT:   %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %indvar

@B = common global i16* null, align 8
@A = common global i16* null, align 8
@C = common global i16* null, align 8

define void @f(i64 %offset) {
entry:
  %a = load i16*, i16** @A, align 8
  %b = load i16*, i16** @B, align 8
  %c = load i16*, i16** @C, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvar = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %indvar
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %indvar
  %loadB = load i16, i16* %arrayidxB, align 2

  %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %indvar
  %loadC = load i16, i16* %arrayidxC, align 2

  %mul = mul i16 %loadB, %loadA
  %mul1 = mul i16 %mul, %loadC

  %idx = add i64 %indvar, %offset
  %arrayidxA2 = getelementptr inbounds i16, i16* %a, i64 %idx
  store i16 %mul1, i16* %arrayidxA2, align 2

  %add = add nuw nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
