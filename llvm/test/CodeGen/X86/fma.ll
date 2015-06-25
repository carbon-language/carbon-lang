; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=+fma,-fma4  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=-fma,-fma4  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mattr=+fma,-fma4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=x86_64-apple-darwin10  -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -mtriple=x86_64-apple-darwin10  -mattr=+avx512f,-fma,-fma4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma4  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FMA-CALL

; CHECK-LABEL: test_f32:
; CHECK-FMA-INST: vfmadd213ss
; CHECK-FMA-CALL: fmaf
define float @test_f32(float %a, float %b, float %c) #0 {
entry:
  %call = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %call
}

; CHECK-LABEL: test_f64:
; CHECK-FMA-INST: vfmadd213sd
; CHECK-FMA-CALL: fma
define double @test_f64(double %a, double %b, double %c) #0 {
entry:
  %call = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %call
}

; CHECK-LABEL: test_f80:
; CHECK: fmal
define x86_fp80 @test_f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) #0 {
entry:
  %call = call x86_fp80 @llvm.fma.f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c)
  ret x86_fp80 %call
}

; CHECK-LABEL: test_f32_cst:
; CHECK-NOT: vfmadd
define float @test_f32_cst() #0 {
entry:
  %call = call float @llvm.fma.f32(float 3.0, float 3.0, float 3.0)
  ret float %call
}

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)
declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80)

attributes #0 = { nounwind }
