; RUN: opt < %s -cost-model -analyze | FileCheck %s

; The cost model does not have any target information so it just makes boring
; assumptions.

; -- No triple in this module --

; CHECK-LABEL: function 'no_info'
; CHECK: cost of 1 {{.*}} add
; CHECK: cost of 1 {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i32 %arg, %arg
  ret i32 %e
}

define i8 @addressing_mode_reg_reg(i8* %a, i32 %b) {
; CHECK-LABEL: function 'addressing_mode_reg_reg'
  %p = getelementptr i8, i8* %a, i32 %b ; NoTTI accepts reg+reg addressing.
; CHECK: cost of 0 {{.*}} getelementptr
  %v = load i8, i8* %p
  ret i8 %v
}

; CHECK-LABEL: function 'addressing_mode_scaled_reg'
define i32 @addressing_mode_scaled_reg(i32* %a, i32 %b) {
  ; NoTTI rejects reg+scale*reg addressing.
  %p = getelementptr i32, i32* %a, i32 %b
; CHECK: cost of 1 {{.*}} getelementptr
  %v = load i32, i32* %p
  ret i32 %v
}
