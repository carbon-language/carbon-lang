; RUN: opt -simplifycfg -S -o - < %s | FileCheck %s

declare void @helper(i32)

define void @test1(i1 %a, i1 %b) {
; CHECK: @test1
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

;; test5 - The case where it jumps to the default target will be removed.
define void @test5(i32 %M, i32 %N) nounwind uwtable {
entry:
  switch i32 %N, label %sw2 [
    i32 1, label %sw2
    i32 2, label %sw.bb
    i32 3, label %sw.bb1
  ], !prof !3
; CHECK: test5
; CHECK: switch i32 %N, label %sw2 [
; CHECK: i32 3, label %sw.bb1
; CHECK: i32 2, label %sw.bb
; CHECK: ], !prof !2

sw.bb:
  call void @helper(i32 0)
  br label %sw.epilog

sw.bb1:
  call void @helper(i32 1)
  br label %sw.epilog

sw2:
  call void @helper(i32 2)
  br label %sw.epilog

sw.epilog:
  ret void
}

;; test6 - Some cases of the second switch are pruned during optimization.
;; Then the second switch will be converted to a branch, finally, the first
;; switch and the branch will be merged into a single switch.
define void @test6(i32 %M, i32 %N) nounwind uwtable {
entry:
  switch i32 %N, label %sw2 [
    i32 1, label %sw2
    i32 2, label %sw.bb
    i32 3, label %sw.bb1
  ], !prof !4
; CHECK: test6
; CHECK: switch i32 %N, label %sw.epilog
; CHECK: i32 3, label %sw.bb1
; CHECK: i32 2, label %sw.bb
; CHECK: i32 4, label %sw.bb5
; CHECK: ], !prof !3

sw.bb:
  call void @helper(i32 0)
  br label %sw.epilog

sw.bb1:
  call void @helper(i32 1)
  br label %sw.epilog

sw2:
;; Here "case 2" is invalidated since the default case of the first switch
;; does not include "case 2".
  switch i32 %N, label %sw.epilog [
    i32 2, label %sw.bb4
    i32 4, label %sw.bb5
  ], !prof !5

sw.bb4:
  call void @helper(i32 2)
  br label %sw.epilog

sw.bb5:
  call void @helper(i32 3)
  br label %sw.epilog

sw.epilog:
  ret void
}

;; This test is based on test1 but swapped the targets of the second branch.
define void @test1_swap(i1 %a, i1 %b) {
; CHECK: @test1_swap
entry:
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %or.cond, label %Y, label %Z, !prof !4

X:
  %c = or i1 %b, false
  br i1 %c, label %Y, label %Z, !prof !1

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
!3 = metadata !{metadata !"branch_weights", i32 4, i32 3, i32 2, i32 1}
!4 = metadata !{metadata !"branch_weights", i32 4, i32 3, i32 2, i32 1}
!5 = metadata !{metadata !"branch_weights", i32 7, i32 6, i32 5}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 5, i32 11}
; CHECK: !1 = metadata !{metadata !"branch_weights", i32 1, i32 5}
; CHECK: !2 = metadata !{metadata !"branch_weights", i32 7, i32 1, i32 2}
; CHECK: !3 = metadata !{metadata !"branch_weights", i32 49, i32 12, i32 24, i32 35}
; CHECK: !4 = metadata !{metadata !"branch_weights", i32 11, i32 5}
; CHECK-NOT: !5
