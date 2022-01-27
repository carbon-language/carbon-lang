; RUN: opt -passes=vector-combine -S %s | FileCheck %s

; Negative test for extract + cmp + binop - don't try this with scalable vectors.
; Moved from X86/extract-cmp-binop.ll

define i1 @scalable(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @scalable(
; CHECK-NEXT:    [[E1:%.*]] = extractelement <vscale x 4 x i32> [[A:%.*]], i32 3
; CHECK-NEXT:    [[E2:%.*]] = extractelement <vscale x 4 x i32> [[A]], i32 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[E1]], 42
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i32 [[E2]], -8
; CHECK-NEXT:    [[R:%.*]] = xor i1 [[CMP1]], [[CMP2]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %e1 = extractelement <vscale x 4 x i32> %a, i32 3
  %e2 = extractelement <vscale x 4 x i32> %a, i32 1
  %cmp1 = icmp sgt i32 %e1, 42
  %cmp2 = icmp sgt i32 %e2, -8
  %r = xor i1 %cmp1, %cmp2
  ret i1 %r
}
