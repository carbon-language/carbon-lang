; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=ALL
;
; INNERMOST:    Function: f
; INNERMOST:    Region: %bb9---%bb18
; INNERMOST:    Max Loop Depth:  1
; INNERMOST:    Context:
; INNERMOST:    [p_0] -> {  : p_0 >= -2199023255552 and p_0 <= 2199023254528 }
; INNERMOST:    Assumed Context:
; INNERMOST:    [p_0] -> {  :  }
; INNERMOST:    p0: {0,+,(sext i32 %N to i64)}<%bb3>
; INNERMOST:    Alias Groups (0):
; INNERMOST:        n/a
; INNERMOST:    Statements {
; INNERMOST:      Stmt_bb12
; INNERMOST:            Domain :=
; INNERMOST:                [p_0] -> { Stmt_bb12[i0] : i0 >= 0 and p_0 >= 1 and i0 <= -1 + p_0 };
; INNERMOST:            Schedule :=
; INNERMOST:                [p_0] -> { Stmt_bb12[i0] -> [i0] };
; INNERMOST:            ReadAccess := [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [p_0] -> { Stmt_bb12[i0] -> MemRef_A[i0] };
; INNERMOST:            MustWriteAccess :=  [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [p_0] -> { Stmt_bb12[i0] -> MemRef_A[i0] };
; INNERMOST:    }
;
; ALL:    Function: f
; ALL:    Region: %bb3---%bb20
; ALL:    Max Loop Depth:  1
; ALL:    Context:
; ALL:    {  :  }
; ALL:    Assumed Context:
; ALL:    {  :  }
; ALL:    Alias Groups (0):
; ALL:        n/a
; ALL:    Statements {
; ALL:      Stmt_(bb4 => bb18)
; ALL:            Domain :=
; ALL:                { Stmt_(bb4 => bb18)[i0] : i0 >= 0 and i0 <= 1023 };
; ALL:            Schedule :=
; ALL:                { Stmt_(bb4 => bb18)[i0] -> [i0] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb4 => bb18)[i0] -> MemRef_A[i0] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb4 => bb18)[i0] -> MemRef_A[o0] : o0 <= 2199023254526 and o0 >= 0 };
; ALL:            MayWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb4 => bb18)[i0] -> MemRef_A[o0] : o0 <= 2199023254526 and o0 >= 0 };
; ALL:    }
;
;    void f(int *A, int N) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i])
;          for (int j = 0; j < N * i; j++)
;            A[j]++;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb3

bb3:                                              ; preds = %bb19, %bb
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb19 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv1, 1024
  br i1 %exitcond, label %bb4, label %bb20

bb4:                                              ; preds = %bb3
  %tmp5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp6 = load i32, i32* %tmp5, align 4
  %tmp7 = icmp eq i32 %tmp6, 0
  br i1 %tmp7, label %bb18, label %bb8

bb8:                                              ; preds = %bb4
  br label %bb9

bb9:                                              ; preds = %bb16, %bb8
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb16 ], [ 0, %bb8 ]
  %tmp10 = mul nsw i64 %indvars.iv1, %tmp
  %tmp11 = icmp slt i64 %indvars.iv, %tmp10
  br i1 %tmp11, label %bb12, label %bb17

bb12:                                             ; preds = %bb9
  %tmp13 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp14 = load i32, i32* %tmp13, align 4
  %tmp15 = add nsw i32 %tmp14, 1
  store i32 %tmp15, i32* %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb9

bb17:                                             ; preds = %bb9
  br label %bb18

bb18:                                             ; preds = %bb4, %bb17
  br label %bb19

bb19:                                             ; preds = %bb18
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb3

bb20:                                             ; preds = %bb3
  ret void
}
