; RUN: lli -O0 -force-interpreter < %s

; libffi does not support fp128 so we don’t test it
declare float  @llvm.sin.f32(float)
declare double @llvm.sin.f64(double)
declare float  @llvm.cos.f32(float)
declare double @llvm.cos.f64(double)
declare float  @llvm.ceil.f32(float)
declare double @llvm.ceil.f64(double)

define i32 @main() {
  %sin32 = call float @llvm.sin.f32(float 0.000000e+00)
  %sin64 = call double @llvm.sin.f64(double 0.000000e+00)
  %cos32 = call float @llvm.cos.f32(float 0.000000e+00)
  %cos64 = call double @llvm.cos.f64(double 0.000000e+00)
  %ceil32 = call float @llvm.ceil.f32(float 0.000000e+00)
  %ceil64 = call double @llvm.ceil.f64(double 0.000000e+00)
  ret i32 0
}
