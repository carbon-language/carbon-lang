; RUN: llc < %s -march=x86 -mattr=+sse41,-avx | FileCheck %s

; PR11674
define void @fpext_frommem(<2 x float>* %in, <2 x double>* %out) {
entry:
; TODO: We should be able to generate cvtps2pd for the load.
; For now, just check that we generate something sane.
; CHECK: cvtss2sd
; CHECK: cvtss2sd
  %0 = load <2 x float>* %in, align 8
  %1 = fpext <2 x float> %0 to <2 x double>
  store <2 x double> %1, <2 x double>* %out, align 1
  ret void
}
