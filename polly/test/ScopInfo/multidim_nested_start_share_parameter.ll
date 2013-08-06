; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo(long n, long m, long o, double A[n][m][o]) {
;
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++) {
;         A[i+3][j-4][k+7] = 1.0;
;         A[i+13][j-14][k+17] = 11.0;
;       }
; }
;
; Access function:
;
;   {{{(56 + (8 * (-4 + (3 * %m)) * %o) + %A),+,(8 * %m * %o)}<%for.i>,+,
;      (8 * %o)}<%for.j>,+,8}<%for.k>
;   {{{(136 + (8 * (-14 + (13 * %m)) * %o) + %A),+,(8 * %m * %o)}<%for.i>,+,
;      (8 * %o)}<%for.j>,+,8}<%for.k>
;
; They should share the following parameters:
;     p1: {0,+,(8 * %o)}<%for.j>
;     p2: {0,+,(8 * %m * %o)}<%for.i>
;

define void @foo(i64 %n, i64 %m, i64 %o, double* %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j.inc ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j ], [ %k.inc, %for.k.inc ]

  %offset0 = add nsw i64 %i, 3
  %subscript0 = mul i64 %offset0, %m
  %offset1 = add nsw i64 %j, -4
  %subscript1 = add i64 %offset1, %subscript0
  %subscript2 = mul i64 %subscript1, %o
  %offset2 = add nsw i64 %k, 7
  %subscript3 = add i64 %subscript2, %offset2
  %idx = getelementptr inbounds double* %A, i64 %subscript3
  store double 1.0, double* %idx

  %offset3 = add nsw i64 %i, 13
  %subscript4 = mul i64 %offset3, %m
  %offset4 = add nsw i64 %j, -14
  %subscript5 = add i64 %offset4, %subscript4
  %subscript6 = mul i64 %subscript5, %o
  %offset5 = add nsw i64 %k, 17
  %subscript7 = add i64 %subscript6, %offset5
  %idx1 = getelementptr inbounds double* %A, i64 %subscript7
  store double 11.0, double* %idx1

  br label %for.k.inc

for.k.inc:
  %k.inc = add nsw i64 %k, 1
  %k.exitcond = icmp eq i64 %k.inc, %o
  br i1 %k.exitcond, label %for.j.inc, label %for.k

for.j.inc:
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, %m
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, %n
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}

; CHECK: p0: %o
; CHECK: p1: {0,+,(8 * %o)}<%for.j>
; CHECK: p2: {0,+,(8 * %m * %o)}<%for.i>
; CHECK: p3: (8 * (-4 + (3 * %m)) * %o)
; CHECK: p4: (8 * (-14 + (13 * %m)) * %o)
; CHECK-NOT: p4

; CHECK:   [o, p_1, p_2, p_3, p_4] -> { Stmt_for_k[i0] -> MemRef_A[o0] : 8o0 = 56 + p_1 + p_2 + p_3 + 8i0 };
; CHECK:   [o, p_1, p_2, p_3, p_4] -> { Stmt_for_k[i0] -> MemRef_A[o0] : 8o0 = 136 + p_1 + p_2 + p_4 + 8i0 };

