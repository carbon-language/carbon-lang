; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:     -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr | FileCheck %s

define void @eieio_test() {
; CHECK-LABEL: @eieio_test
; CHECK: ori r2, r2, 0
; CHECK-NEXT: ori r2, r2, 0
; CHECK-NEXT: eieio
; CHECK-NEXT: blr

entry:
  tail call void @llvm.ppc.eieio()
  ret void
}

declare void @llvm.ppc.eieio()
