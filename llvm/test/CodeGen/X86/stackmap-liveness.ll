; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -disable-fp-elim | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -disable-fp-elim -enable-stackmap-liveness| FileCheck -check-prefix=LIVE %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; CHECK-NEXT:   .long   0
; Num LargeConstants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   3

; CHECK-LABEL:  .long L{{.*}}-_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; LIVE-LABEL:   .long L{{.*}}-_liveness
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  2
; LIVE-NEXT:    .short  7
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 8
; LIVE-NEXT:    .short  19
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 16

; CHECK-LABEL:  .long L{{.*}}-_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; LIVE-LABEL:   .long L{{.*}}-_liveness
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  6
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 2
; LIVE-NEXT:    .short  7
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 8
; LIVE-NEXT:    .short  8
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 8
; LIVE-NEXT:    .short  17
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 32
; LIVE-NEXT:    .short  18
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 32
; LIVE-NEXT:    .short  19
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 16

; CHECK-LABEL:  .long L{{.*}}-_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; LIVE-LABEL:   .long L{{.*}}-_liveness
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  0
; LIVE-NEXT:    .short  2
; LIVE-NEXT:    .short  7
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 8
; LIVE-NEXT:    .short  19
; LIVE-NEXT:    .byte 0
; LIVE-NEXT:    .byte 16
define void @liveness() {
entry:
  %a1 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
  call void (i32, i32, ...)* @llvm.experimental.stackmap(i32 1, i32 5)
  %a2 = call i64 asm sideeffect "", "={r8}"() nounwind
  %a3 = call i8 asm sideeffect "", "={ah}"() nounwind
  %a4 = call <4 x double> asm sideeffect "", "={ymm0}"() nounwind
  %a5 = call <4 x double> asm sideeffect "", "={ymm1}"() nounwind
  call void (i32, i32, ...)* @llvm.experimental.stackmap(i32 2, i32 5)
  call void asm sideeffect "", "{r8},{ah},{ymm0},{ymm1}"(i64 %a2, i8 %a3, <4 x double> %a4, <4 x double> %a5) nounwind
  call void (i32, i32, ...)* @llvm.experimental.stackmap(i32 3, i32 5)
  call void asm sideeffect "", "{xmm2}"(<2 x double> %a1) nounwind
  ret void
}

declare void @llvm.experimental.stackmap(i32, i32, ...)
