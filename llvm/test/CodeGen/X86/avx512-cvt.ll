; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s

; CHECK-LABEL: sitof32
; CHECK: vcvtdq2ps %zmm
; CHECK: ret
define <16 x float> @sitof32(<16 x i32> %a) nounwind {
  %b = sitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %b
}

; CHECK-LABEL: fptosi00
; CHECK: vcvttps2dq %zmm
; CHECK: ret
define <16 x i32> @fptosi00(<16 x float> %a) nounwind {
  %b = fptosi <16 x float> %a to <16 x i32>
  ret <16 x i32> %b
}

; CHECK-LABEL: fptoui00
; CHECK: vcvttps2udq
; CHECK: ret
define <16 x i32> @fptoui00(<16 x float> %a) nounwind {
  %b = fptoui <16 x float> %a to <16 x i32>
  ret <16 x i32> %b
}

; CHECK-LABEL: fptoui_256
; CHECK: vcvttps2udq
; CHECK: ret
define <8 x i32> @fptoui_256(<8 x float> %a) nounwind {
  %b = fptoui <8 x float> %a to <8 x i32>
  ret <8 x i32> %b
}

; CHECK-LABEL: fptoui_128
; CHECK: vcvttps2udq
; CHECK: ret
define <4 x i32> @fptoui_128(<4 x float> %a) nounwind {
  %b = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %b
}

; CHECK-LABEL: fptoui01
; CHECK: vcvttpd2udq
; CHECK: ret
define <8 x i32> @fptoui01(<8 x double> %a) nounwind {
  %b = fptoui <8 x double> %a to <8 x i32>
  ret <8 x i32> %b
}

; CHECK-LABEL: sitof64
; CHECK: vcvtdq2pd %ymm
; CHECK: ret
define <8 x double> @sitof64(<8 x i32> %a) {
  %b = sitofp <8 x i32> %a to <8 x double>
  ret <8 x double> %b
}

; CHECK-LABEL: fptosi01
; CHECK: vcvttpd2dq %zmm
; CHECK: ret
define <8 x i32> @fptosi01(<8 x double> %a) {
  %b = fptosi <8 x double> %a to <8 x i32>
  ret <8 x i32> %b
}

; CHECK-LABEL: fptrunc00
; CHECK: vcvtpd2ps %zmm
; CHECK-NEXT: vcvtpd2ps %zmm
; CHECK-NEXT: vinsertf64x4    $1
; CHECK: ret
define <16 x float> @fptrunc00(<16 x double> %b) nounwind {
  %a = fptrunc <16 x double> %b to <16 x float>
  ret <16 x float> %a
}

; CHECK-LABEL: fpext00
; CHECK: vcvtps2pd %ymm0, %zmm0
; CHECK: ret
define <8 x double> @fpext00(<8 x float> %b) nounwind {
  %a = fpext <8 x float> %b to <8 x double>
  ret <8 x double> %a
}

; CHECK-LABEL: funcA
; CHECK: vcvtsi2sdq (%rdi){{.*}} encoding: [0x62
; CHECK: ret
define double @funcA(i64* nocapture %e) {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to double
  ret double %conv
}

; CHECK-LABEL: funcB
; CHECK: vcvtsi2sdl (%{{.*}} encoding: [0x62
; CHECK: ret
define double @funcB(i32* %e) {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to double
  ret double %conv
}

; CHECK-LABEL: funcC
; CHECK: vcvtsi2ssl (%{{.*}} encoding: [0x62
; CHECK: ret
define float @funcC(i32* %e) {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to float
  ret float %conv
}

; CHECK-LABEL: i64tof32
; CHECK: vcvtsi2ssq  (%{{.*}} encoding: [0x62
; CHECK: ret
define float @i64tof32(i64* %e) {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to float
  ret float %conv
}

; CHECK-LABEL: fpext
; CHECK: vcvtss2sd {{.*}} encoding: [0x62
; CHECK: ret
define void @fpext() {
entry:
  %f = alloca float, align 4
  %d = alloca double, align 8
  %tmp = load float* %f, align 4
  %conv = fpext float %tmp to double
  store double %conv, double* %d, align 8
  ret void
}

; CHECK-LABEL: fpround_scalar
; CHECK: vmovsd {{.*}} encoding: [0x62
; CHECK: vcvtsd2ss {{.*}} encoding: [0x62
; CHECK: vmovss {{.*}} encoding: [0x62
; CHECK: ret
define void @fpround_scalar() nounwind uwtable {
entry:
  %f = alloca float, align 4
  %d = alloca double, align 8
  %tmp = load double* %d, align 8
  %conv = fptrunc double %tmp to float
  store float %conv, float* %f, align 4
  ret void
}

; CHECK-LABEL: long_to_double
; CHECK: vmovq {{.*}} encoding: [0x62
; CHECK: ret
define double @long_to_double(i64 %x) {
   %res = bitcast i64 %x to double
   ret double %res
}

; CHECK-LABEL: double_to_long
; CHECK: vmovq {{.*}} encoding: [0x62
; CHECK: ret
define i64 @double_to_long(double %x) {
   %res = bitcast double %x to i64
   ret i64 %res
}

; CHECK-LABEL: int_to_float
; CHECK: vmovd {{.*}} encoding: [0x62
; CHECK: ret
define float @int_to_float(i32 %x) {
   %res = bitcast i32 %x to float
   ret float %res
}

; CHECK-LABEL: float_to_int
; CHECK: vmovd {{.*}} encoding: [0x62
; CHECK: ret
define i32 @float_to_int(float %x) {
   %res = bitcast float %x to i32
   ret i32 %res
}

; CHECK-LABEL: uitof64
; CHECK: vcvtudq2pd
; CHECK: vextracti64x4
; CHECK: vcvtudq2pd
; CHECK: ret
define <16 x double> @uitof64(<16 x i32> %a) nounwind {
  %b = uitofp <16 x i32> %a to <16 x double>
  ret <16 x double> %b
}

; CHECK-LABEL: uitof32
; CHECK: vcvtudq2ps
; CHECK: ret
define <16 x float> @uitof32(<16 x i32> %a) nounwind {
  %b = uitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %b
}

; CHECK-LABEL: uitof32_256
; CHECK: vcvtudq2ps
; CHECK: ret
define <8 x float> @uitof32_256(<8 x i32> %a) nounwind {
  %b = uitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %b
}

; CHECK-LABEL: uitof32_128
; CHECK: vcvtudq2ps
; CHECK: ret
define <4 x float> @uitof32_128(<4 x i32> %a) nounwind {
  %b = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %b
}

; CHECK-LABEL: @fptosi02
; CHECK: vcvttss2si {{.*}} encoding: [0x62
; CHECK: ret
define i32 @fptosi02(float %a) nounwind {
  %b = fptosi float %a to i32
  ret i32 %b
}

; CHECK-LABEL: @fptoui02
; CHECK: vcvttss2usi {{.*}} encoding: [0x62
; CHECK: ret
define i32 @fptoui02(float %a) nounwind {
  %b = fptoui float %a to i32
  ret i32 %b
}

; CHECK-LABEL: @uitofp02
; CHECK: vcvtusi2ss
; CHECK: ret
define float @uitofp02(i32 %a) nounwind {
  %b = uitofp i32 %a to float
  ret float %b
}

; CHECK-LABEL: @uitofp03
; CHECK: vcvtusi2sd
; CHECK: ret
define double @uitofp03(i32 %a) nounwind {
  %b = uitofp i32 %a to double
  ret double %b
}
