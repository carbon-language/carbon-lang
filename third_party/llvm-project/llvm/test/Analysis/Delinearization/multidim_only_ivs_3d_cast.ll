; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s
; void foo(int n, int m, int o, double A[n][m][o]) {
;
;   for (int i = 0; i < n; i++)
;     for (int j = 0; j < m; j++)
;       for (int k = 0; k < o; k++)
;         A[i][j][k] = 1.0;
; }

; AddRec: {{{%A,+,(8 * (zext i32 %m to i64) * (zext i32 %o to i64))}<%for.i>,+,(8 * (zext i32 %o to i64))}<%for.j>,+,8}<%for.k>
; CHECK: Base offset: %A
; CHECK: ArrayDecl[UnknownSize][(zext i32 %m to i64)][(zext i32 %o to i64)] with elements of 8 bytes.
; CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %n, i32 %m, i32 %o, double* %A) {
entry:
  %m_zext = zext i32 %m to i64
  %n_zext = zext i32 %o to i64
  br label %for.i

for.i:
  %i = phi i64 [ %i.inc, %for.i.inc ], [ 0, %entry ]
  br label %for.j

for.j:
  %j = phi i64 [ %j.inc, %for.j.inc ], [ 0, %for.i ]
  br label %for.k

for.k:
  %k = phi i64 [ %k.inc, %for.k.inc ], [ 0, %for.j ]
  %tmp = mul i64 %i, %m_zext
  %tmp1 = trunc i64 %j to i32
  %tmp2 = trunc i64 %i to i32
  %mul.us.us = mul nsw i32 %tmp1, %tmp2
  %tmp.us.us = add i64 %j, %tmp
  %tmp17.us.us = mul i64 %tmp.us.us, %n_zext
  %subscript = add i64 %tmp17.us.us, %k
  %idx = getelementptr inbounds double, double* %A, i64 %subscript
  store double 1.0, double* %idx
  br label %for.k.inc

for.k.inc:
  %k.inc = add i64 %k, 1
  %k.inc.trunc = trunc i64 %k.inc to i32
  %k.exitcond = icmp eq i32 %k.inc.trunc, %o
  br i1 %k.exitcond, label %for.j.inc, label %for.k

for.j.inc:
  %j.inc = add i64 %j, 1
  %j.inc.trunc = trunc i64 %j.inc to i32
  %j.exitcond = icmp eq i32 %j.inc.trunc, %m
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add i64 %i, 1
  %i.inc.trunc = trunc i64 %i.inc to i32
  %i.exitcond = icmp eq i32 %i.inc.trunc, %n
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}
