; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s --check-prefix=IR
;
; CHECK:          Invariant Accesses: {
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [N, tmp1] -> { Stmt_for_body[i0] -> MemRef_I[0] };
; CHECK-NEXT:             Execution Context: [N, tmp1] -> {  : N > 0 and (tmp1 >= 43 or tmp1 <= 41) }
; CHECK-NEXT:     }
; CHECK:          Invalid Context:
; CHECK-NEXT:     [N, tmp1] -> {  : tmp1 = 42 and N > 0 }
;
; IR:       polly.preload.begin:
; IR-NEXT:    br i1 false, label %polly.start, label %for.cond
;
;    void f(int *A, int *I, int N) {
;      for (int i = 0; i < N; i++) {
;        if (*I == 42)
;          *I = 0;
;        else
;          A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %I, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp1 = load i32, i32* %I, align 4
  %cmp1 = icmp eq i32 %tmp1, 42
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  store i32 0, i32* %I, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp2, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
