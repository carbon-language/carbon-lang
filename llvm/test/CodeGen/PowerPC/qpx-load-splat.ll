; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

; Function Attrs: norecurse nounwind readonly
define <4 x double> @foo(double* nocapture readonly %a) #0 {
entry:
  %0 = load double, double* %a, align 8, !tbaa !1
  %vecinit.i = insertelement <4 x double> undef, double %0, i32 0
  %shuffle.i = shufflevector <4 x double> %vecinit.i, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %shuffle.i

; CHECK-LABEL: @foo
; CHECK: lfd 1, 0(3)
; CHECK: blr
}

define <4 x double> @foox(double* nocapture readonly %a, i64 %idx) #0 {
entry:
  %p = getelementptr double, double* %a, i64 %idx
  %0 = load double, double* %p, align 8, !tbaa !1
  %vecinit.i = insertelement <4 x double> undef, double %0, i32 0
  %shuffle.i = shufflevector <4 x double> %vecinit.i, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %shuffle.i

; CHECK-LABEL: @foox
; CHECK: sldi [[REG1:[0-9]+]], 4, 3
; CHECK: lfdx 1, 3, [[REG1]]
; CHECK: blr
}

define <4 x double> @fooxu(double* nocapture readonly %a, i64 %idx, double** %pptr) #0 {
entry:
  %p = getelementptr double, double* %a, i64 %idx
  %0 = load double, double* %p, align 8, !tbaa !1
  %vecinit.i = insertelement <4 x double> undef, double %0, i32 0
  %shuffle.i = shufflevector <4 x double> %vecinit.i, <4 x double> undef, <4 x i32> zeroinitializer
  store double* %p, double** %pptr, align 8
  ret <4 x double> %shuffle.i

; CHECK-LABEL: @foox
; CHECK: sldi [[REG1:[0-9]+]], 4, 3
; CHECK: lfdux 1, 3, [[REG1]]
; CHECK: std 3, 0(5)
; CHECK: blr
}

define <4 x float> @foof(float* nocapture readonly %a) #0 {
entry:
  %0 = load float, float* %a, align 4, !tbaa !1
  %vecinit.i = insertelement <4 x float> undef, float %0, i32 0
  %shuffle.i = shufflevector <4 x float> %vecinit.i, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %shuffle.i

; CHECK-LABEL: @foof
; CHECK: lfs 1, 0(3)
; CHECK: blr
}

define <4 x float> @foofx(float* nocapture readonly %a, i64 %idx) #0 {
entry:
  %p = getelementptr float, float* %a, i64 %idx
  %0 = load float, float* %p, align 4, !tbaa !1
  %vecinit.i = insertelement <4 x float> undef, float %0, i32 0
  %shuffle.i = shufflevector <4 x float> %vecinit.i, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %shuffle.i

; CHECK-LABEL: @foofx
; CHECK: sldi [[REG1:[0-9]+]], 4, 2
; CHECK: lfsx 1, 3, [[REG1]]
; CHECK: blr
}

attributes #0 = { norecurse nounwind readonly "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="a2q" "target-features"="+qpx,-altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"bgclang r264510-20160326 clang version 3.9.0  (based on LLVM 3.9.0svn)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}

