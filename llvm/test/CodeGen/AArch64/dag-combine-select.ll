; RUN: llc -o - %s | FileCheck %s
target triple = "arm64--"

; Ensure that we transform select(C0, x, select(C1, x, y)) towards
; select(C0 | C1, x, y) so we can use CMP;CCMP for the implementation.
; CHECK-LABEL: test0:
; CHECK: cmp w0, #7
; CHECK: ccmp w1, #0, #0, ne
; CHECK: csel w0, w1, w2, gt
; CHECK: ret
define i32 @test0(i32 %v0, i32 %v1, i32 %v2) {
  %cmp1 = icmp eq i32 %v0, 7
  %cmp2 = icmp sgt i32 %v1, 0
  %sel0 = select i1 %cmp1, i32 %v1, i32 %v2
  %sel1 = select i1 %cmp2, i32 %v1, i32 %sel0
  ret i32 %sel1
}
