; RUN: opt -simplifycfg -S -o - < %s | FileCheck %s

declare void @helper(i32)

define void @test1(i1 %a, i1 %b) {
; CHECK @test1
entry:
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %or.cond, label %Z, label %Y, !prof !0

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

!0 = metadata !{metadata !"branch_weights", i32 1, i32 2}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 2, i32 1}
