; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vpermilps
define <8 x float> @funcA(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 1, i32 2, i32 3, i32 1, i32 5, i32 6, i32 7, i32 5>
  ret <8 x float> %shuffle
}

; CHECK: vpermilpd
define <4 x double> @funcB(<4 x double> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 3>
  ret <4 x double> %shuffle
}

