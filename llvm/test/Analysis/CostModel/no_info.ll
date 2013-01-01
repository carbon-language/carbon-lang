; RUN: opt < %s -cost-model -analyze | FileCheck %s

; The cost model does not have any target information so it can't make a decision.

; -- No triple in this module --

;CHECK: Unknown cost {{.*}} add
;CHECK: Unknown cost {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i32 %arg, %arg
  ret i32 %e
}
