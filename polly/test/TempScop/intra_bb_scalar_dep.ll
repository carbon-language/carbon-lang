; RUN: opt %loadPolly -basicaa -polly-analyze-ir -disable-polly-intra-scop-scalar-to-array -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-analyze-ir  -polly-codegen-scev -disable-polly-intra-scop-scalar-to-array -analyze < %s | FileCheck %s

; void f(long A[], int N, int *init_ptr) {
;   long i, j;
;
;   for (i = 0; i < N; ++i) {
;     for (i = 0; i < N; ++i) {
;       init = *init_ptr;
;       A[i] = init + 2;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) #0 {
entry:
  br label %for.i

for.i:                                            ; preds = %for.i.end, %entry
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:                                       ; preds = %for.i
  br label %for.j

for.j:                                            ; preds = %for.j, %entry.next
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %init = load i64* %init_ptr
  %init_plus_two = add i64 %init, 2
  %scevgep = getelementptr i64* %A, i64 %indvar.j
  store i64 %init_plus_two, i64* %scevgep
; CHECK: BB: for.j
; CHECK: Read init_ptr[0]
; CHECK: Write A[{0,+,8}<%for.j>]
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:                                        ; preds = %for.j
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:                                           ; preds = %for.i.end
  ret void
}

attributes #0 = { nounwind }
