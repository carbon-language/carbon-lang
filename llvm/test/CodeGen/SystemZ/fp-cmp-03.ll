; Test 128-bit floating-point comparison.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; There is no memory form of 128-bit comparison.
define i64 @f1(i64 %a, i64 %b, fp128 *%ptr, float %f2) {
; CHECK-LABEL: f1:
; CHECK: lxebr %f0, %f0
; CHECK: ld %f1, 0(%r4)
; CHECK: ld %f3, 8(%r4)
; CHECK: cxbr %f1, %f0
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2x = fpext float %f2 to fp128
  %f1 = load fp128 *%ptr
  %cond = fcmp oeq fp128 %f1, %f2x
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
