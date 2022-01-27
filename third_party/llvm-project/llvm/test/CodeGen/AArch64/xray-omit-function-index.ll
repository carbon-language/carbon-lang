; RUN: llc -filetype=asm -no-xray-index -o - -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  ret
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0
; CHECK:       .xword .Lxray_sled_0
; CHECK:       .xword .Lxray_sled_1
; CHECK-LABEL: Lxray_sleds_end0

; CHECK-NOT: xray_fn_idx