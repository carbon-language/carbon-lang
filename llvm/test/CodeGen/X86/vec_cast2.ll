; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx -x86-experimental-vector-widening-legalization | FileCheck %s --check-prefix=CHECK-WIDE

;CHECK-LABEL: foo1_8:
;CHECK: vcvtdq2ps
;CHECK: ret
;
;CHECK-WIDE-LABEL: foo1_8:
;CHECK-WIDE:      vpmovzxbd %xmm0, %xmm1
;CHECK-WIDE-NEXT: vpslld $24, %xmm1, %xmm1
;CHECK-WIDE-NEXT: vpsrad $24, %xmm1, %xmm1
;CHECK-WIDE-NEXT: vpshufb {{.*}}, %xmm0, %xmm0
;CHECK-WIDE-NEXT: vpslld $24, %xmm0, %xmm0
;CHECK-WIDE-NEXT: vpsrad $24, %xmm0, %xmm0
;CHECK-WIDE-NEXT: vinsertf128 $1, %xmm0, %ymm1, %ymm0
;CHECK-WIDE-NEXT: vcvtdq2ps %ymm0, %ymm0
;CHECK-WIDE-NEXT: ret
define <8 x float> @foo1_8(<8 x i8> %src) {
  %res = sitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

;CHECK-LABEL: foo1_4:
;CHECK: vcvtdq2ps
;CHECK: ret
;
;CHECK-WIDE-LABEL: foo1_4:
;CHECK-WIDE:      vpmovzxbd %xmm0, %xmm0
;CHECK-WIDE-NEXT: vpslld $24, %xmm0, %xmm0
;CHECK-WIDE-NEXT: vpsrad $24, %xmm0, %xmm0
;CHECK-WIDE-NEXT: vcvtdq2ps %xmm0, %xmm0
;CHECK-WIDE-NEXT: ret
define <4 x float> @foo1_4(<4 x i8> %src) {
  %res = sitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

;CHECK-LABEL: foo2_8:
;CHECK: vcvtdq2ps
;CHECK: ret
;
;CHECK-WIDE-LABEL: foo2_8:
;CHECK-WIDE: vcvtdq2ps %ymm{{.*}}, %ymm{{.*}}
;CHECK-WIDE: ret
define <8 x float> @foo2_8(<8 x i8> %src) {
  %res = uitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

;CHECK-LABEL: foo2_4:
;CHECK: vcvtdq2ps
;CHECK: ret
;
;CHECK-WIDE-LABEL: foo2_4:
;CHECK-WIDE: vcvtdq2ps %xmm{{.*}}, %xmm{{.*}}
;CHECK-WIDE: ret
define <4 x float> @foo2_4(<4 x i8> %src) {
  %res = uitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

;CHECK-LABEL: foo3_8:
;CHECK: vcvttps2dq
;CHECK: ret
define <8 x i8> @foo3_8(<8 x float> %src) {
  %res = fptosi <8 x float> %src to <8 x i8>
  ret <8 x i8> %res
}
;CHECK-LABEL: foo3_4:
;CHECK: vcvttps2dq
;CHECK: ret
define <4 x i8> @foo3_4(<4 x float> %src) {
  %res = fptosi <4 x float> %src to <4 x i8>
  ret <4 x i8> %res
}

