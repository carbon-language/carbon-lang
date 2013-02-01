; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@stored_label = global i8* null

define void @foo() {
; CHECK: foo:
  %lab = load i8** @stored_label
  indirectbr i8* %lab, [label  %otherlab, label %retlab]
; CHECK: adrp {{x[0-9]+}}, stored_label
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, #:lo12:stored_label]
; CHECK: br {{x[0-9]+}}

otherlab:
  ret void
retlab:
  ret void
}
