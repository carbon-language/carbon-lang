; RUN: opt %loadPolly -basicaa -polly-scops -analyze < %s | FileCheck %s

; void f(long A[], int N, int *init_ptr) {
;   long i, j;
;
;   for (i = 0; i < N; ++i) {
;     init = *init_ptr;
;     for (i = 0; i < N; ++i) {
;       A[i] = init + 2;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; CHECK-LABEL: Function: f
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         MemRef_init_ptr[0]
; CHECK-NEXT:     Execution Context: [N] -> {  : N >= 1 or N <= -1 }

define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) nounwind {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:
; CHECK-NOT:      Stmt_entry_next
  %init = load i64, i64* %init_ptr
  br label %for.j

for.j:
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
; CHECK:      Stmt_for_j
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:         [N] -> { Stmt_for_j[i0, i1] -> MemRef_init[] };
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         [N] -> { Stmt_for_j[i0, i1] -> MemRef_A[i1] };
  %init_plus_two = add i64 %init, 2
  %scevgep = getelementptr i64, i64* %A, i64 %indvar.j
  store i64 %init_plus_two, i64* %scevgep
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:
  ret void
}

; CHECK-LABEL: Function: g
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         MemRef_init_ptr[0]
; CHECK-NEXT:     Execution Context: [N] -> {  : N >= 1 or N <= -1 }
define void @g(i64* noalias %A, i64 %N, i64* noalias %init_ptr) nounwind {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:
; CHECK-NOT:      Stmt_entry_next
  %init = load i64, i64* %init_ptr
  br label %for.j

for.j:
; CHECK:      Stmt_for_j
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar.j
  store i64 %init, i64* %scevgep
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:         [N] -> { Stmt_for_j[i0, i1] -> MemRef_init[] };
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         [N] -> { Stmt_for_j[i0, i1] -> MemRef_A[i1] };
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:
  ret void
}
