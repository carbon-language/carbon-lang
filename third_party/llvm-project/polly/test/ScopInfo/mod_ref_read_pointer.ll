; RUN: opt %loadPolly -basic-aa -polly-scops -analyze -polly-allow-modref-calls \
; RUN:  < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -polly-codegen -disable-output \
; RUN: -polly-allow-modref-calls < %s
;
; Check that we assume the call to func has a read on the whole A array.
;
; CHECK:      Stmt_for_body
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_for_body[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:   MustWriteAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[2 + i0] };
; CHECK-NEXT:   ReadAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[o0] };
;
;    #pragma readonly
;    int func(int *A);
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i + 2] = func(A);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %call = call i32 @func(i32* %A) #2
  %tmp = add nsw i64 %i, 2
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp
  store i32 %call, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

declare i32 @func(i32*) #0

attributes #0 = { nounwind readonly }
