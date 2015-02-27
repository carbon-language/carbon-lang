; RUN: llc < %s -march=x86 -mattr=+sse4.1,-avx | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=corei7-avx | FileCheck --check-prefix=AVX %s

; PR11674
define void @fpext_frommem(<2 x float>* %in, <2 x double>* %out) {
; CHECK-LABEL: fpext_frommem:
; AVX-LABEL: fpext_frommem:
entry:
; CHECK: cvtps2pd (%{{.+}}), %xmm{{[0-9]+}}
; AVX: vcvtps2pd (%{{.+}}), %xmm{{[0-9]+}}
  %0 = load <2 x float>, <2 x float>* %in, align 8
  %1 = fpext <2 x float> %0 to <2 x double>
  store <2 x double> %1, <2 x double>* %out, align 1
  ret void
}

define void @fpext_frommem4(<4 x float>* %in, <4 x double>* %out) {
; CHECK-LABEL: fpext_frommem4:
; AVX-LABEL: fpext_frommem4:
entry:
; CHECK: cvtps2pd (%{{.+}}), %xmm{{[0-9]+}}
; CHECK: cvtps2pd 8(%{{.+}}), %xmm{{[0-9]+}}
; AVX: vcvtps2pd (%{{.+}}), %ymm{{[0-9]+}}
  %0 = load <4 x float>, <4 x float>* %in
  %1 = fpext <4 x float> %0 to <4 x double>
  store <4 x double> %1, <4 x double>* %out, align 1
  ret void
}

define void @fpext_frommem8(<8 x float>* %in, <8 x double>* %out) {
; CHECK-LABEL: fpext_frommem8:
; AVX-LABEL: fpext_frommem8:
entry:
; CHECK: cvtps2pd (%{{.+}}), %xmm{{[0-9]+}}
; CHECK: cvtps2pd 8(%{{.+}}), %xmm{{[0-9]+}}
; CHECK: cvtps2pd 16(%{{.+}}), %xmm{{[0-9]+}}
; CHECK: cvtps2pd 24(%{{.+}}), %xmm{{[0-9]+}}
; AVX: vcvtps2pd 16(%{{.+}}), %ymm{{[0-9]+}}
; AVX: vcvtps2pd (%{{.+}}), %ymm{{[0-9]+}}
  %0 = load <8 x float>, <8 x float>* %in
  %1 = fpext <8 x float> %0 to <8 x double>
  store <8 x double> %1, <8 x double>* %out, align 1
  ret void
}
