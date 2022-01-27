; Test strict conversion of floating-point values to unsigned integers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128, metadata)

declare i64 @llvm.experimental.constrained.fptoui.i64.f32(float, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128, metadata)

; Test f32->i32.
define i32 @f1(float %f) #0 {
; CHECK-LABEL: f1:
; CHECK: clfebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) #0 {
; CHECK-LABEL: f2:
; CHECK: clfdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(fp128 *%src) #0 {
; CHECK-LABEL: f3:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clfxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f32->i64.
define i64 @f4(float %f) #0 {
; CHECK-LABEL: f4:
; CHECK: clgebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f64->i64.
define i64 @f5(double %f) #0 {
; CHECK-LABEL: f5:
; CHECK: clgdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f128->i64.
define i64 @f6(fp128 *%src) #0 {
; CHECK-LABEL: f6:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clgxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

attributes #0 = { strictfp }
