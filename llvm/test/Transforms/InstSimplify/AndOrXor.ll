; RUN: opt < %s -instsimplify -S | FileCheck %s

define i64 @pow2(i32 %x) {
; CHECK-LABEL: @pow2(
  %negx = sub i32 0, %x
  %x2 = and i32 %x, %negx
  %e = zext i32 %x2 to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}

define i64 @pow2b(i32 %x) {
; CHECK-LABEL: @pow2b(
  %sh = shl i32 2, %x
  %e = zext i32 %sh to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}

define i32 @sub_neg_nuw(i32 %x, i32 %y) {
; CHECK-LABEL: @sub_neg_nuw(
  %neg = sub nuw i32 0, %y
  %sub = sub i32 %x, %neg
  ret i32 %sub
; CHECK: ret i32 %x
}
