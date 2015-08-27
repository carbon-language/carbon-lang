; RUN: llc -march=x86-64 -print-machineinstrs=expand-isel-pseudos %s -o /dev/null 2>&1 | FileCheck %s


; CHECK-LABEL: test

define void @test(i32 %x) nounwind {
entry:
  switch i32 %x, label %sw.default [
    i32 54, label %sw.bb
    i32 55, label %sw.bb
    i32 56, label %sw.bb
    i32 58, label %sw.bb
    i32 67, label %sw.bb
    i32 68, label %sw.bb
    i32 134, label %sw.bb
    i32 140, label %sw.bb
  ], !prof !1

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.default:
  call void @foo(i32 1)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: BB#4:
; CHECK: Successors according to CFG: BB#1(10) BB#6(10)
; CHECK: BB#6:
; CHECK: Successors according to CFG: BB#1(10) BB#2(10)
}

declare void @foo(i32)

!1 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10} 
