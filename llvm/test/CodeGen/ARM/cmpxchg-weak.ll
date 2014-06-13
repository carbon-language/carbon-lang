; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s

define void @test_cmpxchg_weak(i32 *%addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak:

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  %oldval = extractvalue { i32, i1 } %pair, 0
; CHECK:     dmb ish
; CHECK:     ldrex   [[LOADED:r[0-9]+]], [r0]
; CHECK:     cmp     [[LOADED]], r1
; CHECK:     strexeq [[SUCCESS:r[0-9]+]], r2, [r0]
; CHECK:     cmpeq   [[SUCCESS]], #0
; CHECK:     bne     [[DONE:LBB[0-9]+_[0-9]+]]
; CHECK:     dmb     ish
; CHECK: [[DONE]]:
; CHECK:     str     r3, [r0]
; CHECK:     bx      lr

  store i32 %oldval, i32* %addr
  ret void
}


define i1 @test_cmpxchg_weak_to_bool(i32, i32 *%addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak_to_bool:

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  %success = extractvalue { i32, i1 } %pair, 1

; CHECK:      dmb     ish
; CHECK:      mov     r0, #0
; CHECK:      ldrex   [[LOADED:r[0-9]+]], [r1]
; CHECK:      cmp     [[LOADED]], r2
; CHECK:      strexeq [[STATUS:r[0-9]+]], r3, [r1]
; CHECK:      cmpeq   [[STATUS]], #0
; CHECK:      bne     [[DONE:LBB[0-9]+_[0-9]+]]
; CHECK:      dmb     ish
; CHECK:      mov     r0, #1
; CHECK: [[DONE]]:
; CHECK:      bx      lr

  ret i1 %success
}
