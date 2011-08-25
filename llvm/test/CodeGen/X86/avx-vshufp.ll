; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vshufps  $-53, %ymm
define <8 x float> @A(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  ret <8 x float> %shuffle
}

; CHECK: vshufpd  $10, %ymm
define <4 x double> @B(<4 x double> %a, <4 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x double> %shuffle
}

; CHECK: vshufps  $-53, %ymm
define <8 x float> @C(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 undef, i32 undef, i32 11, i32 undef, i32 6, i32 12, i32 undef>
  ret <8 x float> %shuffle
}

; CHECK: vshufpd  $2, %ymm
define <4 x double> @D(<4 x double> %a, <4 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 undef>
  ret <4 x double> %shuffle
}
