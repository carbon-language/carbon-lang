; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vunpckhps
define <8 x float> @unpackhips(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x float> %shuffle.i
}

; CHECK: vunpckhpd
define <4 x double> @unpackhipd(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x double> %shuffle.i
}

; CHECK: vunpcklps
define <8 x float> @unpacklops(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %shuffle.i
}

; CHECK: vunpcklpd
define <4 x double> @unpacklopd(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x double> %shuffle.i
}

; CHECK-NOT: vunpcklps %ymm
define <8 x float> @unpacklops-not(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x float> %shuffle.i
}

; CHECK-NOT: vunpcklpd %ymm
define <4 x double> @unpacklopd-not(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x double> %shuffle.i
}

; CHECK-NOT: vunpckhps %ymm
define <8 x float> @unpackhips-not(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %shuffle.i
}

; CHECK-NOT: vunpckhpd %ymm
define <4 x double> @unpackhipd-not(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x double> %shuffle.i
}

;;;;
;;;; Unpack versions using the fp unit for int unpacking
;;;;

; CHECK: vunpckhps
define <8 x i32> @unpackhips1(<8 x i32> %src1, <8 x i32> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x i32> %src1, <8 x i32> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpckhps (%
define <8 x i32> @unpackhips2(<8 x i32>* %src1, <8 x i32>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <8 x i32>* %src1
  %b = load <8 x i32>* %src2
  %shuffle.i = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpckhpd
define <4 x i64> @unpackhipd1(<4 x i64> %src1, <4 x i64> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x i64> %src1, <4 x i64> %src2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpckhpd (%
define <4 x i64> @unpackhipd2(<4 x i64>* %src1, <4 x i64>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <4 x i64>* %src1
  %b = load <4 x i64>* %src2
  %shuffle.i = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpcklps
define <8 x i32> @unpacklops1(<8 x i32> %src1, <8 x i32> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x i32> %src1, <8 x i32> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpcklps (%
define <8 x i32> @unpacklops2(<8 x i32>* %src1, <8 x i32>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <8 x i32>* %src1
  %b = load <8 x i32>* %src2
  %shuffle.i = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpcklpd
define <4 x i64> @unpacklopd1(<4 x i64> %src1, <4 x i64> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x i64> %src1, <4 x i64> %src2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpcklpd (%
define <4 x i64> @unpacklopd2(<4 x i64>* %src1, <4 x i64>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <4 x i64>* %src1
  %b = load <4 x i64>* %src2
  %shuffle.i = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i64> %shuffle.i
}
