; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine -polly-allow-nonaffine-branches -disable-polly-intra-scop-scalar-to-array=false -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine -polly-allow-nonaffine-branches -disable-polly-intra-scop-scalar-to-array=true -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s -check-prefix=SCALAR
;
; CHECK:    Function: f
; CHECK:    Region: %bb1---%bb13
; CHECK:    Max Loop Depth:  1
; CHECK:    Context:
; CHECK:    {  :  }
; CHECK:    Assumed Context:
; CHECK:    {  :  }
; CHECK:    Alias Groups (0):
; CHECK:        n/a
; CHECK:    Statements {
; CHECK:      Stmt_(bb3 => bb11)
; CHECK:            Domain :=
; CHECK:                { Stmt_(bb3 => bb11)[i0] : i0 >= 0 and i0 <= 1023 };
; CHECK:            Schedule :=
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> [i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> MemRef_C[i0] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> MemRef_tmp4_s2a[0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> MemRef_tmp4_s2a[0] };
; CHECK:            ReadAccess := [Reduction Type: +] [Scalar: 0]
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= -2147483648 };
; CHECK:            MayWriteAccess := [Reduction Type: +] [Scalar: 0]
; CHECK:                { Stmt_(bb3 => bb11)[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= -2147483648 };
; CHECK:    }

; SCALAR:    Function: f
; SCALAR:    Region: %bb1---%bb13
; SCALAR:    Max Loop Depth:  1
; SCALAR:    Context:
; SCALAR:    {  :  }
; SCALAR:    Assumed Context:
; SCALAR:    {  :  }
; SCALAR:    Alias Groups (0):
; SCALAR:        n/a
; SCALAR:    Statements {
; SCALAR:      Stmt_(bb3 => bb11)
; SCALAR:            Domain :=
; SCALAR:                { Stmt_(bb3 => bb11)[i0] : i0 >= 0 and i0 <= 1023 };
; SCALAR:            Schedule :=
; SCALAR:                { Stmt_(bb3 => bb11)[i0] -> [i0] };
; SCALAR:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; SCALAR:                { Stmt_(bb3 => bb11)[i0] -> MemRef_C[i0] };
; SCALAR:            ReadAccess := [Reduction Type: +] [Scalar: 0]
; SCALAR:                { Stmt_(bb3 => bb11)[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= -2147483648 };
; SCALAR:            MayWriteAccess := [Reduction Type: +] [Scalar: 0]
; SCALAR:                { Stmt_(bb3 => bb11)[i0] -> MemRef_A[o0] : o0 <= 2147483645 and o0 >= -2147483648 };
; SCALAR:    }

;
;    void f(int * restrict A, int * restrict C) {
;      int j;
;      for (int i = 0; i < 1024; i++) {
;        while ((j = C[i]))
;          A[j]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias %A, i32* noalias %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb12 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb13

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb6, %bb2
  %tmp = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %tmp4 = load i32, i32* %tmp, align 4
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb11, label %bb6

bb6:                                              ; preds = %bb3
  %tmp7 = sext i32 %tmp4 to i64
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %tmp7
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, 1
  store i32 %tmp10, i32* %tmp8, align 4
  br label %bb3

bb11:                                             ; preds = %bb3
  br label %bb12

bb12:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb13:                                             ; preds = %bb1
  ret void
}
