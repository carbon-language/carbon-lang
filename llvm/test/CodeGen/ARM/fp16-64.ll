; RUN: llc -mtriple=arm -mattr=+fp-armv8 < %s | \
; RUN:   FileCheck --check-prefix=CHECK --check-prefix=V8 %s
; RUN: llc -mtriple=arm -mattr=+vfp3,+d16 < %s | \
; RUN:   FileCheck --check-prefix=CHECK --check-prefix=NOV8 %s

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone

define void @vcvt_f64_f16(i16* %x, double* %y) nounwind {
entry:
; CHECK-LABEL: vcvt_f64_f16
  %0 = load i16, i16* %x, align 2
  %1 = tail call float @llvm.convert.from.fp16.f32(i16 %0)
  %conv = fpext float %1 to double
; CHECK-V8: vcvtb.f64.f16
; CHECK-NOV8-NOT: vcvtb.f64.f16
  store double %conv, double* %y, align 8
  ret void
}

define void @vcvt_f16_f64(i16* %x, double* %y) nounwind {
entry:
; CHECK-LABEL: vcvt_f16_f64
  %0 = load double, double* %y, align 8
  %conv = fptrunc double %0 to float
; CHECK-V8: vcvtb.f16.f64
; CHECK-NOV8-NOT: vcvtb.f16.f64
  %1 = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  store i16 %1, i16* %x, align 2
  ret void
}
