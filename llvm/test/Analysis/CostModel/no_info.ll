; RUN: opt < %s -cost-model -analyze | FileCheck %s

; The cost model does not have any target information so it just makes boring
; assumptions.

; -- No triple in this module --

;CHECK: cost of 1 {{.*}} add
;CHECK: cost of 1 {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i32 %arg, %arg
  ret i32 %e
}
