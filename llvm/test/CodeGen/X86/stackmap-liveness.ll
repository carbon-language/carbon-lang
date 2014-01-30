; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -disable-fp-elim | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -disable-fp-elim -enable-stackmap-liveness| FileCheck -check-prefix=STACK %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -disable-fp-elim -enable-patchpoint-liveness| FileCheck -check-prefix=PATCH %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; CHECK-NEXT:   .long 0
; Num Functions
; CHECK-NEXT:   .long 2
; CHECK-NEXT:   .long _stackmap_liveness
; CHECK-NEXT:   .long 8
; CHECK-NEXT:   .long _mixed_liveness
; CHECK-NEXT:   .long 8
; Num LargeConstants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   5
define void @stackmap_liveness() {
entry:
  %a1 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
; StackMap 1 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0

; StackMap 1 (stackmap liveness information enabled)
; STACK-LABEL:  .long L{{.*}}-_stackmap_liveness
; STACK-NEXT:   .short  0
; STACK-NEXT:   .short  0
; Num LiveOut Entries: 2
; STACK-NEXT:   .short  2
; LiveOut Entry 1: %RSP (8 bytes)
; STACK-NEXT:   .short  7
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 8
; LiveOut Entry 2: %YMM2 (16 bytes) --> %XMM2
; STACK-NEXT:   .short  19
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 16

; StackMap 1 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 0
; PATCH-NEXT:   .short  0
  call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 1, i32 5)
  %a2 = call i64 asm sideeffect "", "={r8}"() nounwind
  %a3 = call i8 asm sideeffect "", "={ah}"() nounwind
  %a4 = call <4 x double> asm sideeffect "", "={ymm0}"() nounwind
  %a5 = call <4 x double> asm sideeffect "", "={ymm1}"() nounwind

; StackMap 2 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0

; StackMap 2 (stackmap liveness information enabled)
; STACK-LABEL:  .long L{{.*}}-_stackmap_liveness
; STACK-NEXT:   .short  0
; STACK-NEXT:   .short  0
; Num LiveOut Entries: 6
; STACK-NEXT:   .short  6
; LiveOut Entry 2: %RAX (1 bytes) --> %AL or %AH
; STACK-NEXT:   .short  0
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 1
; LiveOut Entry 2: %RSP (8 bytes)
; STACK-NEXT:   .short  7
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 8
; LiveOut Entry 2: %R8 (8 bytes)
; STACK-NEXT:   .short  8
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 8
; LiveOut Entry 2: %YMM0 (32 bytes)
; STACK-NEXT:   .short  17
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 32
; LiveOut Entry 2: %YMM1 (32 bytes)
; STACK-NEXT:   .short  18
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 32
; LiveOut Entry 2: %YMM2 (16 bytes) --> %XMM2
; STACK-NEXT:   .short  19
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 16

; StackMap 2 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 0
; PATCH-NEXT:   .short  0
  call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 2, i32 5)
  call void asm sideeffect "", "{r8},{ah},{ymm0},{ymm1}"(i64 %a2, i8 %a3, <4 x double> %a4, <4 x double> %a5) nounwind

; StackMap 3 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0

; StackMap 3 (stackmap liveness information enabled)
; STACK-LABEL:  .long L{{.*}}-_stackmap_liveness
; STACK-NEXT:   .short  0
; STACK-NEXT:   .short  0
; Num LiveOut Entries: 2
; STACK-NEXT:   .short  2
; LiveOut Entry 2: %RSP (8 bytes)
; STACK-NEXT:   .short  7
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 8
; LiveOut Entry 2: %YMM2 (16 bytes) --> %XMM2
; STACK-NEXT:   .short  19
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 16

; StackMap 3 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 0
; PATCH-NEXT:   .short  0
  call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 3, i32 5)
  call void asm sideeffect "", "{xmm2}"(<2 x double> %a1) nounwind
  ret void
}

define void @mixed_liveness() {
entry:
  %a1 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
; StackMap 4 (stackmap liveness information enabled)
; STACK-LABEL:  .long L{{.*}}-_mixed_liveness
; STACK-NEXT:   .short  0
; STACK-NEXT:   .short  0
; Num LiveOut Entries: 1
; STACK-NEXT:   .short  1
; LiveOut Entry 1: %YMM2 (16 bytes) --> %XMM2
; STACK-NEXT:   .short  19
; STACK-NEXT:   .byte 0
; STACK-NEXT:   .byte 16
; StackMap 5 (stackmap liveness information enabled)
; STACK-LABEL:  .long L{{.*}}-_mixed_liveness
; STACK-NEXT:   .short  0
; STACK-NEXT:   .short  0
; Num LiveOut Entries: 0
; STACK-NEXT:   .short  0

; StackMap 4 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_mixed_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 0
; PATCH-NEXT:   .short  0
; StackMap 5 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_mixed_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 2
; PATCH-NEXT:   .short  2
; LiveOut Entry 1: %RSP (8 bytes)
; PATCH-NEXT:   .short  7
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 8
; LiveOut Entry 1: %YMM2 (16 bytes) --> %XMM2
; PATCH-NEXT:   .short  19
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 16
  call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 4, i32 5)
  call anyregcc void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 5, i32 0, i8* null, i32 0)
  call void asm sideeffect "", "{xmm2}"(<2 x double> %a1) nounwind
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
