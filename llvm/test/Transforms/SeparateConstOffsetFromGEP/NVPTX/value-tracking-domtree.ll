; RUN: opt < %s -separate-const-offset-from-gep -value-tracking-dom-conditions -reassociate-geps-verify-no-dead-code -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "nvptx64-unknown-unknown"

; if (i == 4)
;   p = &input[i | 3];
;
; =>
;
; if (i == 4) {
;   base = &input[i];
;   p = &base[3];
; }
;
; We should treat (i | 3) as (i + 3) because i is guaranteed to be 4, which
; does not share any set bits with 3.
define float* @guarded_or(float* %input, i64 %i) {
; CHECK-LABEL: @guarded_or(
entry:
  %is4 = icmp eq i64 %i, 4
  br i1 %is4, label %then, label %exit

then:
  %or = or i64 %i, 3
  %p = getelementptr inbounds float, float* %input, i64 %or
; CHECK: [[base:[^ ]+]] = getelementptr float, float* %input, i64 %i
; CHECK: getelementptr inbounds float, float* [[base]], i64 3
  ret float* %p

exit:
  ret float* null
}
