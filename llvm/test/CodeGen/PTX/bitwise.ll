; RUN: llc < %s -march=ptx32 | FileCheck %s

; preds

define ptx_device i32 @t1_and_preds(i1 %x, i1 %y) {
; CHECK: and.pred %p{{[0-9]+}}, %p{{[0-9]+}}, %p{{[0-9]+}}
  %c = and i1 %x, %y
  %d = zext i1 %c to i32 
  ret i32 %d
}

define ptx_device i32 @t1_or_preds(i1 %x, i1 %y) {
; CHECK: or.pred %p{{[0-9]+}}, %p{{[0-9]+}}, %p{{[0-9]+}}
  %a = or i1 %x, %y
  %b = zext i1 %a to i32 
  ret i32 %b
}

define ptx_device i32 @t1_xor_preds(i1 %x, i1 %y) {
; CHECK: xor.pred %p{{[0-9]+}}, %p{{[0-9]+}}, %p{{[0-9]+}}
  %a = xor i1 %x, %y
  %b = zext i1 %a to i32 
  ret i32 %b
}
