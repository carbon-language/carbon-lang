; Test 64-bit floating-point subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare double @foo()

; Check register subtraction.
define double @f1(double %f1, double %f2) {
; CHECK-LABEL: f1:
; CHECK: sdbr %f0, %f2
; CHECK: br %r14
  %res = fsub double %f1, %f2
  ret double %res
}

; Check the low end of the SDB range.
define double @f2(double %f1, double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: sdb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double *%ptr
  %res = fsub double %f1, %f2
  ret double %res
}

; Check the high end of the aligned SDB range.
define double @f3(double %f1, double *%base) {
; CHECK-LABEL: f3:
; CHECK: sdb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 511
  %f2 = load double *%ptr
  %res = fsub double %f1, %f2
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, double *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: sdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 512
  %f2 = load double *%ptr
  %res = fsub double %f1, %f2
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, double *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: sdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 -1
  %f2 = load double *%ptr
  %res = fsub double %f1, %f2
  ret double %res
}

; Check that SDB allows indices.
define double @f6(double %f1, double *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: sdb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double *%base, i64 %index
  %ptr2 = getelementptr double *%ptr1, i64 100
  %f2 = load double *%ptr2
  %res = fsub double %f1, %f2
  ret double %res
}

; Check that subtractions of spilled values can use SDB rather than SDBR.
define double @f7(double *%ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: sdb %f0, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr double *%ptr0, i64 2
  %ptr2 = getelementptr double *%ptr0, i64 4
  %ptr3 = getelementptr double *%ptr0, i64 6
  %ptr4 = getelementptr double *%ptr0, i64 8
  %ptr5 = getelementptr double *%ptr0, i64 10
  %ptr6 = getelementptr double *%ptr0, i64 12
  %ptr7 = getelementptr double *%ptr0, i64 14
  %ptr8 = getelementptr double *%ptr0, i64 16
  %ptr9 = getelementptr double *%ptr0, i64 18
  %ptr10 = getelementptr double *%ptr0, i64 20

  %val0 = load double *%ptr0
  %val1 = load double *%ptr1
  %val2 = load double *%ptr2
  %val3 = load double *%ptr3
  %val4 = load double *%ptr4
  %val5 = load double *%ptr5
  %val6 = load double *%ptr6
  %val7 = load double *%ptr7
  %val8 = load double *%ptr8
  %val9 = load double *%ptr9
  %val10 = load double *%ptr10

  %ret = call double @foo()

  %sub0 = fsub double %ret, %val0
  %sub1 = fsub double %sub0, %val1
  %sub2 = fsub double %sub1, %val2
  %sub3 = fsub double %sub2, %val3
  %sub4 = fsub double %sub3, %val4
  %sub5 = fsub double %sub4, %val5
  %sub6 = fsub double %sub5, %val6
  %sub7 = fsub double %sub6, %val7
  %sub8 = fsub double %sub7, %val8
  %sub9 = fsub double %sub8, %val9
  %sub10 = fsub double %sub9, %val10

  ret double %sub10
}
