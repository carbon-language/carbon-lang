; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcvtdq2ps %ymm
define <8 x float> @funcA(<8 x i32> %a) nounwind {
  %b = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %b
}

; CHECK: vcvttps2dq %ymm
define <8 x i32> @funcB(<8 x float> %a) nounwind {
  %b = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %b
}

; CHECK: vcvtpd2psy %ymm
; CHECK-NEXT: vcvtpd2psy %ymm
; CHECK-NEXT: vinsertf128 $1
define <8 x float> @funcC(<8 x double> %b) nounwind {
  %a = fptrunc <8 x double> %b to <8 x float>
  ret <8 x float> %a
}
