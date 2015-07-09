; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=ALL
;
; INNERMOST:    Function: f
; INNERMOST:    Region: %bb9---%bb17
; INNERMOST:    Max Loop Depth:  1
; INNERMOST:    Context:
; INNERMOST:    [N] -> {  : N >= -2147483648 and N <= 2147483647 }
; INNERMOST:    Assumed Context:
; INNERMOST:    [N] -> {  :  }
; INNERMOST:    p0: %N
; INNERMOST:    Alias Groups (0):
; INNERMOST:        n/a
; INNERMOST:    Statements {
; INNERMOST:      Stmt_bb11
; INNERMOST:            Domain :=
; INNERMOST:                [N] -> { Stmt_bb11[i0] : i0 >= 0 and N >= 1 and i0 <= -1 + N };
; INNERMOST:            Schedule :=
; INNERMOST:                [N] -> { Stmt_bb11[i0] -> [i0] };
; INNERMOST:            ReadAccess := [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [N] -> { Stmt_bb11[i0] -> MemRef_A[i0] };
; INNERMOST:            MustWriteAccess :=  [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [N] -> { Stmt_bb11[i0] -> MemRef_A[i0] };
; INNERMOST:    }
;
; ALL:    Function: f
; ALL:    Region: %bb3---%bb19
; ALL:    Max Loop Depth:  1
; ALL:    Context:
; ALL:    {  :  }
; ALL:    Assumed Context:
; ALL:    {  :  }
; ALL:    Alias Groups (0):
; ALL:        n/a
; ALL:    Statements {
; ALL:      Stmt_bb4__TO__bb17
; ALL:            Domain :=
; ALL:                { Stmt_bb4__TO__bb17[i0] : i0 >= 0 and i0 <= 1023 };
; ALL:            Schedule :=
; ALL:                { Stmt_bb4__TO__bb17[i0] -> [i0] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_bb4__TO__bb17[i0] -> MemRef_A[i0] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_bb4__TO__bb17[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= 0 };
; ALL:            MayWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_bb4__TO__bb17[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= 0 };
; ALL:    }
;
;    void f(int *A, int N) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i])
;          for (int j = 0; j < N; j++)
;            A[j]++;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb3

bb3:                                              ; preds = %bb18, %bb
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb18 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv1, 1024
  br i1 %exitcond, label %bb4, label %bb19

bb4:                                              ; preds = %bb3
  %tmp5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp6 = load i32, i32* %tmp5, align 4
  %tmp7 = icmp eq i32 %tmp6, 0
  br i1 %tmp7, label %bb17, label %bb8

bb8:                                              ; preds = %bb4
  br label %bb9

bb9:                                              ; preds = %bb15, %bb8
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb15 ], [ 0, %bb8 ]
  %tmp10 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp10, label %bb11, label %bb16

bb11:                                             ; preds = %bb9
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = add nsw i32 %tmp13, 1
  store i32 %tmp14, i32* %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb9

bb16:                                             ; preds = %bb9
  br label %bb17

bb17:                                             ; preds = %bb4, %bb16
  br label %bb18

bb18:                                             ; preds = %bb17
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb3

bb19:                                             ; preds = %bb3
  ret void
}
