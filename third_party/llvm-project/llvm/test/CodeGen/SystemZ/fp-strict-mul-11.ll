; Test strict 128-bit floating-point multiplication on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.experimental.constrained.fmul.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

define void @f1(fp128 *%ptr1, fp128 *%ptr2) #0 {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr1
  %f2 = load fp128, fp128 *%ptr2
  %sum = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %sum, fp128 *%ptr1
  ret void
}

define void @f2(double %f1, double %f2, fp128 *%dst) #0 {
; CHECK-LABEL: f2:
; CHECK-DAG: wflld [[REG1:%v[0-9]+]], %f0
; CHECK-DAG: wflld [[REG2:%v[0-9]+]], %f2
; CHECK: wfmxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %f1,
                                               metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %f2,
                                               metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

attributes #0 = { strictfp }
