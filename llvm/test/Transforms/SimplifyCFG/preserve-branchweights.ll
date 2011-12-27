; RUN: opt -simplifycfg -S -o - < %s | FileCheck %s

declare void @helper(i32)

define void @test1(i1 %a, i1 %b) {
; CHECK @test1
entry:
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %or.cond, label %Z, label %Y, !prof !0

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !1

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test2(i1 %a, i1 %b) {
; CHECK: @test2
entry:
  br i1 %a, label %X, label %Y, !prof !1
; CHECK: br i1 %or.cond, label %Z, label %Y, !prof !1
; CHECK-NOT: !prof

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !2

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test3(i1 %a, i1 %b) {
; CHECK: @test3
; CHECK-NOT: !prof
entry:
  br i1 %a, label %X, label %Y, !prof !1

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test4(i1 %a, i1 %b) {
; CHECK: @test4
; CHECK-NOT: !prof
entry:
  br i1 %a, label %X, label %Y

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !1

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

!0 = metadata !{metadata !"branch_weights", i32 3, i32 5}
!1 = metadata !{metadata !"branch_weights", i32 1, i32 1}
!2 = metadata !{metadata !"branch_weights", i32 1, i32 2}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 5, i32 11}
; CHECK: !1 = metadata !{metadata !"branch_weights", i32 1, i32 5}
; CHECK-NOT: !2
