; RUN: opt < %s -instcombine -S | FileCheck %s

; When shrinking demanded constant operand of an add instruction, keep in
; mind the opcode can be changed to sub and the constant negated. Make sure
; the shrinking the constant would actually reduce the width.
; rdar://11793464

define i64 @t(i64 %key, i64* %val) nounwind {
entry:
; CHECK: @t
; CHECK-NOT: add i64 %0, 2305843009213693951
; CHECK: add i64 %0, -1
  %shr = lshr i64 %key, 3
  %0 = load i64* %val, align 8
  %sub = sub i64 %0, 1
  %and = and i64 %sub, %shr
  ret i64 %and
}
