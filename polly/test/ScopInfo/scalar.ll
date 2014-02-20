; RUN: opt %loadPolly -polly-scops -analyze -disable-polly-intra-scop-scalar-to-array < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* %a, i64 %N) {
entry:
  br label %for

for:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.backedge ]
  br label %S1

S1:
  %scevgep1 = getelementptr i64* %a, i64 %indvar
  %val = load i64* %scevgep1, align 8
  br label %S2

S2:
  %scevgep2 = getelementptr i64* %a, i64 %indvar
  store i64 %val, i64* %scevgep2, align 8
  br label %for.backedge

for.backedge:
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for

return:
  ret void
}

; CHECK: Stmt_S1
; CHECK:       Domain :=
; CHECK:           [N] -> { Stmt_S1[i0] : i0 >= 0 and i0 <= -1 + N };
; CHECK:       Scattering :=
; CHECK:           [N] -> { Stmt_S1[i0] -> scattering[0, i0, 0] };
; CHECK:       ReadAccess :=
; CHECK:           [N] -> { Stmt_S1[i0] -> MemRef_a[i0] };
; CHECK:       MustWriteAccess :=
; CHECK:           [N] -> { Stmt_S1[i0] -> MemRef_val[0] };
; CHECK: Stmt_S2
; CHECK:       Domain :=
; CHECK:           [N] -> { Stmt_S2[i0] : i0 >= 0 and i0 <= -1 + N };
; CHECK:       Scattering :=
; CHECK:           [N] -> { Stmt_S2[i0] -> scattering[0, i0, 1] };
; CHECK:       ReadAccess :=
; CHECK:           [N] -> { Stmt_S2[i0] -> MemRef_val[0] };
; CHECK:       MustWriteAccess :=
; CHECK:           [N] -> { Stmt_S2[i0] -> MemRef_a[i0] };
