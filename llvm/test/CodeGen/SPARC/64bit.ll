; RUN: llc < %s -march=sparcv9 | FileCheck %s

; CHECK: ret2:
; CHECK: or %g0, %i1, %i0
define i64 @ret2(i64 %a, i64 %b) {
  ret i64 %b
}
