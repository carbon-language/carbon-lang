; Test 32-bit floating-point comparison.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @foo()

; Check comparison with registers.
define i64 @f1(i64 %a, i64 %b, float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: cebr %f0, %f2
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the low end of the CEB range.
define i64 @f2(i64 %a, i64 %b, float %f1, float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the high end of the aligned CEB range.
define i64 @f3(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f3:
; CHECK: ceb %f0, 4092(%r4)
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r4, 4096
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check negative displacements, which also need separate address logic.
define i64 @f5(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r4, -4
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that CEB allows indices.
define i64 @f6(i64 %a, i64 %b, float %f1, float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r5, 2
; CHECK: ceb %f0, 400(%r1,%r4)
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr1 = getelementptr float *%base, i64 %index
  %ptr2 = getelementptr float *%ptr1, i64 100
  %f2 = load float *%ptr2
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that comparisons of spilled values can use CEB rather than CEBR.
define float @f7(float *%ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: ceb {{%f[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float *%ptr0, i64 2
  %ptr2 = getelementptr float *%ptr0, i64 4
  %ptr3 = getelementptr float *%ptr0, i64 6
  %ptr4 = getelementptr float *%ptr0, i64 8
  %ptr5 = getelementptr float *%ptr0, i64 10
  %ptr6 = getelementptr float *%ptr0, i64 12
  %ptr7 = getelementptr float *%ptr0, i64 14
  %ptr8 = getelementptr float *%ptr0, i64 16
  %ptr9 = getelementptr float *%ptr0, i64 18
  %ptr10 = getelementptr float *%ptr0, i64 20

  %val0 = load float *%ptr0
  %val1 = load float *%ptr1
  %val2 = load float *%ptr2
  %val3 = load float *%ptr3
  %val4 = load float *%ptr4
  %val5 = load float *%ptr5
  %val6 = load float *%ptr6
  %val7 = load float *%ptr7
  %val8 = load float *%ptr8
  %val9 = load float *%ptr9
  %val10 = load float *%ptr10

  %ret = call float @foo()

  %cmp0 = fcmp olt float %ret, %val0
  %cmp1 = fcmp olt float %ret, %val1
  %cmp2 = fcmp olt float %ret, %val2
  %cmp3 = fcmp olt float %ret, %val3
  %cmp4 = fcmp olt float %ret, %val4
  %cmp5 = fcmp olt float %ret, %val5
  %cmp6 = fcmp olt float %ret, %val6
  %cmp7 = fcmp olt float %ret, %val7
  %cmp8 = fcmp olt float %ret, %val8
  %cmp9 = fcmp olt float %ret, %val9
  %cmp10 = fcmp olt float %ret, %val10

  %sel0 = select i1 %cmp0, float %ret, float 0.0
  %sel1 = select i1 %cmp1, float %sel0, float 1.0
  %sel2 = select i1 %cmp2, float %sel1, float 2.0
  %sel3 = select i1 %cmp3, float %sel2, float 3.0
  %sel4 = select i1 %cmp4, float %sel3, float 4.0
  %sel5 = select i1 %cmp5, float %sel4, float 5.0
  %sel6 = select i1 %cmp6, float %sel5, float 6.0
  %sel7 = select i1 %cmp7, float %sel6, float 7.0
  %sel8 = select i1 %cmp8, float %sel7, float 8.0
  %sel9 = select i1 %cmp9, float %sel8, float 9.0
  %sel10 = select i1 %cmp10, float %sel9, float 10.0

  ret float %sel10
}

; Check comparison with zero.
define i64 @f8(i64 %a, i64 %b, float %f) {
; CHECK-LABEL: f8:
; CHECK: ltebr %f0, %f0
; CHECK-NEXT: je
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq float %f, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
