; RUN: opt < %s -bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i64* @alloc(i32, i8, i32)
declare i32* @alloc2(i32, i32)

; CHECK: @f1
define void @f1(i32 %x) {
  %call = tail call i32* @alloc2(i32 %x, i32 4) nounwind, !alloc_size !0
; CHECK: trap
  store i32 3, i32* %call, align 4
  ret void
}

; CHECK: @f2
define void @f2() {
  %call1 = tail call i32* @alloc2(i32 2, i32 4) nounwind, !alloc_size !0
  %arrayidx = getelementptr i32* %call1, i64 2
; CHECK: br label
  store i32 3, i32* %arrayidx, align 4
  ret void
}

; CHECK: @f3
define void @f3(i32 %x, i8 %y) {
  %call = tail call i64* @alloc(i32 %x, i8 %y, i32 7) nounwind, !alloc_size !1
; CHECK: trap
  store i64 27, i64* %call, align 4
  ret void
}

; CHECK: @f4
define void @f4() {
  %call1 = tail call i32* @alloc2(i32 2, i32 4) nounwind, !alloc_size !0
  %arrayidx = getelementptr i32* %call1, i64 1
; CHECK-NOT: trap
  store i32 3, i32* %arrayidx, align 4
; CHECK: ret
  ret void
}

!0 = metadata !{i32 0, i32 1}
!1 = metadata !{i32 2}
