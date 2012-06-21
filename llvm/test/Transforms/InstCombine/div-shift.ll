; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @t1(i16 zeroext %x, i32 %y) nounwind {
entry:
; CHECK: t1
; CHECK-NOT: sdiv
; CHECK: lshr i32 %conv
  %conv = zext i16 %x to i32
  %s = shl i32 2, %y
  %d = sdiv i32 %conv, %s
  ret i32 %d
}

; rdar://11721329
define i64 @t2(i64 %x, i32 %y) nounwind  {
; CHECK: t2
; CHECK-NOT: udiv
; CHECK: lshr i64 %x
  %1 = shl i32 1, %y
  %2 = zext i32 %1 to i64
  %3 = udiv i64 %x, %2
  ret i64 %3
}
