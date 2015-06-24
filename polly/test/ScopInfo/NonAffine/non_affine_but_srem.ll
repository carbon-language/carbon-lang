; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void pos(float *A, long n) {
;      for (long i = 0; i < 100; i++)
;        A[n % 42] += 1;
;    }
;
; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] :
; CHECK:          exists (e0 = floor((-n + o0)/42):
; CHECK:                  42e0 = -n + o0 and o0 <= 41 and o0 >= 0) };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] :
; CHECK:          exists (e0 = floor((-n + o0)/42):
; CHECK:                  42e0 = -n + o0 and o0 <= 41 and o0 >= 0) };
;
;    void neg(float *A, long n) {
;      for (long i = 0; i < 100; i++)
;        A[n % (-42)] += 1;
;    }
;
; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] :
; CHECK:          exists (e0 = floor((-n + o0)/42):
; CHECK:                  42e0 = -n + o0 and o0 <= 41 and o0 >= 0) };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] :
; CHECK:          exists (e0 = floor((-n + o0)/42):
; CHECK:                  42e0 = -n + o0 and o0 <= 41 and o0 >= 0) };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @pos(float* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, 42
  %tmp3 = getelementptr inbounds float, float* %A, i64 %tmp
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

define void @neg(float* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, -42
  %tmp3 = getelementptr inbounds float, float* %A, i64 %tmp
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
