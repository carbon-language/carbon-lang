; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; CHECK-label: test_or
; CHECK:       cbnz w0, {{LBB[0-9]+_2}}
; CHECK:       cbz w1, {{LBB[0-9]+_1}}
define i64 @test_or(i32 %a, i32 %b) {
bb1:
  %0 = icmp eq i32 %a, 0
  %1 = icmp eq i32 %b, 0
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %bb3, label %bb4, !prof !0

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

; CHECK-label: test_ans
; CHECK:       cbz w0, {{LBB[0-9]+_2}}
; CHECK:       cbnz w1, {{LBB[0-9]+_3}}
define i64 @test_and(i32 %a, i32 %b) {
bb1:
  %0 = icmp ne i32 %a, 0
  %1 = icmp ne i32 %b, 0
  %or.cond = and i1 %0, %1
  br i1 %or.cond, label %bb4, label %bb3, !prof !1

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

declare i64 @bar()

!0 = !{!"branch_weights", i32 5128, i32 32}
!1 = !{!"branch_weights", i32 1024, i32 4136}
