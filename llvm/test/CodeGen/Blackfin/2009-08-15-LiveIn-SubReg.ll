; RUN: llc < %s -march=bfin -verify-machineinstrs

; When joining live intervals of sub-registers, an MBB live-in list is not
; updated properly. The register scavenger asserts on an undefined register.

define i32 @foo(i8 %bar) {
entry:
  switch i8 %bar, label %bb1203 [
    i8 117, label %bb1204
    i8 85, label %bb1204
    i8 106, label %bb1204
  ]

bb1203:                                           ; preds = %entry
  ret i32 1

bb1204:                                           ; preds = %entry, %entry, %entry
  ret i32 2
}
