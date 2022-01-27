; RUN: llc -mtriple armv7-eabi -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv6m-eabi -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-eabi -o - %s | FileCheck %s

declare void @llvm.arm.hint(i32) nounwind

define void @hint_nop() {
entry:
  tail call void @llvm.arm.hint(i32 0) nounwind
  ret void
}

; CHECK-LABEL: hint_nop
; CHECK: nop

define void @hint_yield() {
entry:
  tail call void @llvm.arm.hint(i32 1) nounwind
  ret void
}

; CHECK-LABEL: hint_yield
; CHECK: yield

define void @hint_wfe() {
entry:
  tail call void @llvm.arm.hint(i32 2) nounwind
  ret void
}

; CHECK-LABEL: hint_wfe
; CHECK: wfe

define void @hint_wfi() {
entry:
  tail call void @llvm.arm.hint(i32 3) nounwind
  ret void
}

; CHECK-LABEL: hint_wfi
; CHECK: wfi

define void @hint_sev() {
entry:
  tail call void @llvm.arm.hint(i32 4) nounwind
  ret void
}

; CHECK-LABEL: hint_sev
; CHECK: sev

define void @hint_sevl() {
entry:
  tail call void @llvm.arm.hint(i32 5) nounwind
  ret void
}

; CHECK-LABEL: hint_sevl
; CHECK: hint #5

define void @hint_undefined() {
entry:
  tail call void @llvm.arm.hint(i32 8) nounwind
  ret void
}

; CHECK-LABEL: hint_undefined
; CHECK: hint #8

