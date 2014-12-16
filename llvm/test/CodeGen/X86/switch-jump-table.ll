; RUN: llc -mtriple=i686-pc-gnu-linux < %s | FileCheck %s


; An unreachable default destination is replaced with the most popular case label.

define void @sum2(i32 %x, i32* %to) {
; CHECK-LABEL: sum2:
; CHECK: movl 4(%esp), [[REG:%e[a-z]{2}]]
; CHECK: cmpl $3, [[REG]]
; CHECK: jbe .LBB0_1
; CHECK: movl $4
; CHECK: retl
; CHECK-LABEL: .LBB0_1:
; CHECK-NEXT: jmpl *.LJTI0_0(,[[REG]],4)

entry:
  switch i32 %x, label %default [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb4
  ]
bb0:
  store i32 0, i32* %to
  br label %exit
bb1:
  store i32 1, i32* %to
  br label %exit
bb2:
  store i32 2, i32* %to
  br label %exit
bb3:
  store i32 3, i32* %to
  br label %exit
bb4:
  store i32 4, i32* %to
  br label %exit
exit:
  ret void
default:
  unreachable

; The jump table has four entries.
; CHECK-LABEL: .LJTI0_0:
; CHECK-NEXT: .long  .LBB0_2
; CHECK-NEXT: .long  .LBB0_3
; CHECK-NEXT: .long  .LBB0_4
; CHECK-NEXT: .long  .LBB0_5
; CHECK-NOT: .long
}
