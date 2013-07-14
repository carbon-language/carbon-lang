; Test extensions of f32 to f64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register extension.
define double @f1(float %val) {
; CHECK-LABEL: f1:
; CHECK: ldebr %f0, %f0
; CHECK: br %r14
  %res = fpext float %val to double
  ret double %res
}

; Check the low end of the LDEB range.
define double @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %val = load float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check the high end of the aligned LDEB range.
define double @f3(float *%base) {
; CHECK-LABEL: f3:
; CHECK: ldeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %val = load float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %val = load float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %val = load float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check that LDEB allows indices.
define double @f6(float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: ldeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float *%base, i64 %index
  %ptr2 = getelementptr float *%ptr1, i64 100
  %val = load float *%ptr2
  %res = fpext float %val to double
  ret double %res
}

; Test a case where we spill the source of at least one LDEBR.  We want
; to use LDEB if possible.
define void @f7(double *%ptr1, float *%ptr2) {
; CHECK-LABEL: f7:
; CHECK: ldeb {{%f[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile float *%ptr2
  %val1 = load volatile float *%ptr2
  %val2 = load volatile float *%ptr2
  %val3 = load volatile float *%ptr2
  %val4 = load volatile float *%ptr2
  %val5 = load volatile float *%ptr2
  %val6 = load volatile float *%ptr2
  %val7 = load volatile float *%ptr2
  %val8 = load volatile float *%ptr2
  %val9 = load volatile float *%ptr2
  %val10 = load volatile float *%ptr2
  %val11 = load volatile float *%ptr2
  %val12 = load volatile float *%ptr2
  %val13 = load volatile float *%ptr2
  %val14 = load volatile float *%ptr2
  %val15 = load volatile float *%ptr2
  %val16 = load volatile float *%ptr2

  %ext0 = fpext float %val0 to double
  %ext1 = fpext float %val1 to double
  %ext2 = fpext float %val2 to double
  %ext3 = fpext float %val3 to double
  %ext4 = fpext float %val4 to double
  %ext5 = fpext float %val5 to double
  %ext6 = fpext float %val6 to double
  %ext7 = fpext float %val7 to double
  %ext8 = fpext float %val8 to double
  %ext9 = fpext float %val9 to double
  %ext10 = fpext float %val10 to double
  %ext11 = fpext float %val11 to double
  %ext12 = fpext float %val12 to double
  %ext13 = fpext float %val13 to double
  %ext14 = fpext float %val14 to double
  %ext15 = fpext float %val15 to double
  %ext16 = fpext float %val16 to double

  store volatile float %val0, float *%ptr2
  store volatile float %val1, float *%ptr2
  store volatile float %val2, float *%ptr2
  store volatile float %val3, float *%ptr2
  store volatile float %val4, float *%ptr2
  store volatile float %val5, float *%ptr2
  store volatile float %val6, float *%ptr2
  store volatile float %val7, float *%ptr2
  store volatile float %val8, float *%ptr2
  store volatile float %val9, float *%ptr2
  store volatile float %val10, float *%ptr2
  store volatile float %val11, float *%ptr2
  store volatile float %val12, float *%ptr2
  store volatile float %val13, float *%ptr2
  store volatile float %val14, float *%ptr2
  store volatile float %val15, float *%ptr2
  store volatile float %val16, float *%ptr2

  store volatile double %ext0, double *%ptr1
  store volatile double %ext1, double *%ptr1
  store volatile double %ext2, double *%ptr1
  store volatile double %ext3, double *%ptr1
  store volatile double %ext4, double *%ptr1
  store volatile double %ext5, double *%ptr1
  store volatile double %ext6, double *%ptr1
  store volatile double %ext7, double *%ptr1
  store volatile double %ext8, double *%ptr1
  store volatile double %ext9, double *%ptr1
  store volatile double %ext10, double *%ptr1
  store volatile double %ext11, double *%ptr1
  store volatile double %ext12, double *%ptr1
  store volatile double %ext13, double *%ptr1
  store volatile double %ext14, double *%ptr1
  store volatile double %ext15, double *%ptr1
  store volatile double %ext16, double *%ptr1

  ret void
}
