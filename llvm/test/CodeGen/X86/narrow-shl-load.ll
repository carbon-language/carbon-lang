; RUN: llc -march=x86-64 < %s

; DAGCombiner should fold this code in finite time.

; rdar://8606584

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

define void @D() nounwind readnone {
bb.nph:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %bb.nph
  %tmp6 = load i32* undef, align 4
  %and = or i64 undef, undef
  %conv11 = zext i32 undef to i64
  %conv14 = zext i32 %tmp6 to i64
  %shl15 = shl i64 %conv14, 1
  %shl15.masked = and i64 %shl15, 4294967294
  %and17 = or i64 %shl15.masked, %conv11
  %add = add i64 %and17, 1
  %xor = xor i64 %add, %and
  %tmp20 = load i64* undef, align 8
  %add21 = add i64 %xor, %tmp20
  %conv22 = trunc i64 %add21 to i32
  store i32 %conv22, i32* undef, align 4
  br i1 false, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}
