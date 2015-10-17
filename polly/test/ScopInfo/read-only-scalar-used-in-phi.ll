; RUN: opt %loadPolly -analyze -polly-scops \
; RUN: < %s | FileCheck %s
;
;    float foo(float sum, float A[]) {
;
;      for (long i = 0; i < 100; i++)
;        sum += A[i];
;
;      return sum;
;    }

; CHECK: Stmt_next
; CHECK:       Domain :=
; CHECK:           { Stmt_next[] };
; CHECK:       Schedule :=
; CHECK:           { Stmt_next[] -> [0, 0] };
; CHECK:       ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK:           { Stmt_next[] -> MemRef_sum[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           { Stmt_next[] -> MemRef_phisum__phi[] };
; CHECK: Stmt_bb1
; CHECK:       Domain :=
; CHECK:           { Stmt_bb1[i0] : i0 <= 100 and i0 >= 0 };
; CHECK:       Schedule :=
; CHECK:           { Stmt_bb1[i0] -> [1, i0] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           { Stmt_bb1[i0] -> MemRef_phisum__phi[] };
; CHECK:       ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK:           { Stmt_bb1[i0] -> MemRef_phisum__phi[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           { Stmt_bb1[i0] -> MemRef_phisum[] };
; CHECK:       ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:           { Stmt_bb1[i0] -> MemRef_A[i0] };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float %sum, float* %A) {
bb:
  br label %next

next:
  br i1 true, label %bb1, label %bb7

bb1:
  %i = phi i64 [ 0, %next ], [ %i.next, %bb1 ]
  %phisum = phi float [ %sum, %next ], [ %tmp5, %bb1 ]
  %tmp = getelementptr inbounds float, float* %A, i64 %i
  %tmp4 = load float, float* %tmp, align 4
  %tmp5 = fadd float %phisum, %tmp4
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i, 100
  br i1 %exitcond, label %bb1, label %bb7

bb7:
  %phisummerge = phi float [%phisum, %bb1], [0.0, %next]
  ret float %phisummerge
}
