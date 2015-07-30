; PR23510
; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @_Z3fooPml(
; CHECK: lshr <2 x i64>
; CHECK: lshr <2 x i64>

@total = global i64 0, align 8

define void @_Z3fooPml(i64* nocapture %a, i64 %i) {
entry:
  %tmp = load i64, i64* %a, align 8
  %shr = lshr i64 %tmp, 4
  store i64 %shr, i64* %a, align 8
  %arrayidx1 = getelementptr inbounds i64, i64* %a, i64 1
  %tmp1 = load i64, i64* %arrayidx1, align 8
  %shr2 = lshr i64 %tmp1, 4
  store i64 %shr2, i64* %arrayidx1, align 8
  %arrayidx3 = getelementptr inbounds i64, i64* %a, i64 %i
  %tmp2 = load i64, i64* %arrayidx3, align 8
  %tmp3 = load i64, i64* @total, align 8
  %add = add i64 %tmp3, %tmp2
  store i64 %add, i64* @total, align 8
  %tmp4 = load i64, i64* %a, align 8
  %shr5 = lshr i64 %tmp4, 4
  store i64 %shr5, i64* %a, align 8
  %tmp5 = load i64, i64* %arrayidx1, align 8
  %shr7 = lshr i64 %tmp5, 4
  store i64 %shr7, i64* %arrayidx1, align 8
  %tmp6 = load i64, i64* %arrayidx3, align 8
  %tmp7 = load i64, i64* @total, align 8
  %add9 = add i64 %tmp7, %tmp6
  store i64 %add9, i64* @total, align 8
  ret void
}
