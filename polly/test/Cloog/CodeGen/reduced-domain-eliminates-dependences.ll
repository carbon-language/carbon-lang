; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-codegen -polly-vectorizer=polly < %s

; void f(int a[]) {
;  int i;
;  for (i = 0; i < 10; ++i)
;    A[i] = A[i+5];
; }

; In this test case we import a schedule that limits the iteration domain
; to 0 <= i < 5, which makes the loop parallel. Previously we crashed in such
; cases. This test checks that we instead vectorize the loop.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @reduced-domain-eliminates-dependences(i64* %a) {
entry:
  br label %bb

bb:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %bb ]
  %add = add i64 %indvar, 5
  %scevgep.load = getelementptr i64* %a, i64 %add
  %scevgep.store = getelementptr i64* %a, i64 %indvar
  %val = load i64* %scevgep.load
  store i64 %val, i64* %scevgep.store, align 8
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; CHECK: store <4 x i64> %val_p_vec_full, <4 x i64>* %vector_ptr10
