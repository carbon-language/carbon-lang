; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s -preserve-bc-use-list-order

; Check that musttail and tail roundtrip.

declare cc8191 void @t1_callee()
define cc8191 void @t1() {
; CHECK: tail call cc8191 void @t1_callee()
  tail call cc8191 void @t1_callee()
  ret void
}

declare cc8191 void @t2_callee()
define cc8191 void @t2() {
; CHECK: musttail call cc8191 void @t2_callee()
  musttail call cc8191 void @t2_callee()
  ret void
}
