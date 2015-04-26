; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=false -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s --check-prefix=ALL
;
; Here we have a non-affine loop (in the context of the loop nest)
; and also a non-affine access (A[k]). While we can always model the
; innermost loop as a SCoP of depth 1, we can overapproximate the
; innermost loop in the whole loop nest and model A[k] as a non-affine
; access.
;
; INNERMOST:    Function: f
; INNERMOST:    Region: %bb15---%bb26
; INNERMOST:    Max Loop Depth:  1
; INNERMOST:    p0: {0,+,{0,+,-1}<nw><%bb11>}<nw><%bb13>
; INNERMOST:    p1: {0,+,{0,+,1}<nuw><nsw><%bb11>}<nuw><nsw><%bb13>
; INNERMOST:    p2: {0,+,4}<nuw><nsw><%bb11>
; INNERMOST:    p3: {0,+,4}<nuw><nsw><%bb13>
; INNERMOST:    p4: {0,+,{0,+,4}<nuw><nsw><%bb11>}<%bb13>
; INNERMOST:    Alias Groups (0):
; INNERMOST:        n/a
; INNERMOST:    Statements {
; INNERMOST:      Stmt_bb16
; INNERMOST:            Domain :=
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] : (i0 <= 1023 - p_1 and i0 >= 0 and i0 <= 1024 + p_0) or (i0 >= 0 and i0 >= 1025 - p_1 and i0 <= 1024 + p_0) };
; INNERMOST:            Schedule :=
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] -> [i0] };
; INNERMOST:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_2 };
; INNERMOST:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_3 };
; INNERMOST:            ReadAccess := [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_4 + 4i0 };
; INNERMOST:            MustWriteAccess :=  [Reduction Type: +] [Scalar: 0]
; INNERMOST:                [p_0, p_1, p_2, p_3, p_4] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_4 + 4i0 };
; INNERMOST:    }
;
; ALL:    Function: f
; ALL:    Region: %bb11---%bb29
; ALL:    Max Loop Depth:  2
; ALL:    Context:
; ALL:    {  :  }
; ALL:    Assumed Context:
; ALL:    {  :  }
; ALL:    Alias Groups (0):
; ALL:        n/a
; ALL:    Statements {
; ALL:      Stmt_(bb15 => bb25)
; ALL:            Domain :=
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] : i0 >= 0 and i0 <= 1023 and i1 >= 0 and i1 <= 1023 };
; ALL:            Schedule :=
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] -> [i0, i1] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] -> MemRef_A[i0] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] -> MemRef_A[i1] };
; ALL:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] -> MemRef_A[o0] : o0 <= 2305843009213693949 and o0 >= 0 };
; ALL:            MayWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; ALL:                { Stmt_(bb15 => bb25)[i0, i1] -> MemRef_A[o0] : o0 <= 2305843009213693949 and o0 >= 0 };
; ALL:    }
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          for (int k = i *j;  k < 1024; k++)
;            A[k] += A[i] + A[j];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb11

bb11:                                             ; preds = %bb28, %bb
  %indvars.iv8 = phi i64 [ %indvars.iv.next9, %bb28 ], [ 0, %bb ]
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb28 ], [ 0, %bb ]
  %exitcond10 = icmp ne i64 %indvars.iv8, 1024
  br i1 %exitcond10, label %bb12, label %bb29

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb26, %bb12
  %indvars.iv5 = phi i64 [ %indvars.iv.next6, %bb26 ], [ 0, %bb12 ]
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %bb26 ], [ 0, %bb12 ]
  %exitcond7 = icmp ne i64 %indvars.iv5, 1024
  br i1 %exitcond7, label %bb14, label %bb27

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb24, %bb14
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb24 ], [ %indvars.iv3, %bb14 ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb16, label %bb25

bb16:                                             ; preds = %bb15
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv8
  %tmp17 = load i32, i32* %tmp, align 4
  %tmp18 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp19 = load i32, i32* %tmp18, align 4
  %tmp20 = add nsw i32 %tmp17, %tmp19
  %tmp21 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp22 = load i32, i32* %tmp21, align 4
  %tmp23 = add nsw i32 %tmp22, %tmp20
  store i32 %tmp23, i32* %tmp21, align 4
  br label %bb24

bb24:                                             ; preds = %bb16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb15

bb25:                                             ; preds = %bb15
  br label %bb26

bb26:                                             ; preds = %bb25
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, %indvars.iv1
  br label %bb13

bb27:                                             ; preds = %bb13
  br label %bb28

bb28:                                             ; preds = %bb27
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb11

bb29:                                             ; preds = %bb11
  ret void
}
