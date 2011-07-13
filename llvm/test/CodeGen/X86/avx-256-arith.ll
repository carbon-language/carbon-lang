; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vaddpd
define <4 x double> @addpd256(<4 x double> %y, <4 x double> %x) nounwind uwtable readnone ssp {
entry:
  %add.i = fadd <4 x double> %x, %y
  ret <4 x double> %add.i
}

; CHECK: vaddpd LCP{{.*}}(%rip)
define <4 x double> @addpd256fold(<4 x double> %y) nounwind uwtable readnone ssp {
entry:
  %add.i = fadd <4 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <4 x double> %add.i
}

; CHECK: vaddps
define <8 x float> @addps256(<8 x float> %y, <8 x float> %x) nounwind uwtable readnone ssp {
entry:
  %add.i = fadd <8 x float> %x, %y
  ret <8 x float> %add.i
}

; CHECK: vaddps LCP{{.*}}(%rip)
define <8 x float> @addps256fold(<8 x float> %y) nounwind uwtable readnone ssp {
entry:
  %add.i = fadd <8 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000>
  ret <8 x float> %add.i
}

; CHECK: vsubpd
define <4 x double> @subpd256(<4 x double> %y, <4 x double> %x) nounwind uwtable readnone ssp {
entry:
  %sub.i = fsub <4 x double> %x, %y
  ret <4 x double> %sub.i
}

; CHECK: vsubpd (%
define <4 x double> @subpd256fold(<4 x double> %y, <4 x double>* nocapture %x) nounwind uwtable readonly ssp {
entry:
  %tmp2 = load <4 x double>* %x, align 32
  %sub.i = fsub <4 x double> %y, %tmp2
  ret <4 x double> %sub.i
}

; CHECK: vsubps
define <8 x float> @subps256(<8 x float> %y, <8 x float> %x) nounwind uwtable readnone ssp {
entry:
  %sub.i = fsub <8 x float> %x, %y
  ret <8 x float> %sub.i
}

; CHECK: vsubps (%
define <8 x float> @subps256fold(<8 x float> %y, <8 x float>* nocapture %x) nounwind uwtable readonly ssp {
entry:
  %tmp2 = load <8 x float>* %x, align 32
  %sub.i = fsub <8 x float> %y, %tmp2
  ret <8 x float> %sub.i
}

; CHECK: vmulpd
define <4 x double> @mulpd256(<4 x double> %y, <4 x double> %x) nounwind uwtable readnone ssp {
entry:
  %mul.i = fmul <4 x double> %x, %y
  ret <4 x double> %mul.i
}

; CHECK: vmulpd LCP{{.*}}(%rip)
define <4 x double> @mulpd256fold(<4 x double> %y) nounwind uwtable readnone ssp {
entry:
  %mul.i = fmul <4 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <4 x double> %mul.i
}

; CHECK: vmulps
define <8 x float> @mulps256(<8 x float> %y, <8 x float> %x) nounwind uwtable readnone ssp {
entry:
  %mul.i = fmul <8 x float> %x, %y
  ret <8 x float> %mul.i
}

; CHECK: vmulps LCP{{.*}}(%rip)
define <8 x float> @mulps256fold(<8 x float> %y) nounwind uwtable readnone ssp {
entry:
  %mul.i = fmul <8 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000>
  ret <8 x float> %mul.i
}

; CHECK: vdivpd
define <4 x double> @divpd256(<4 x double> %y, <4 x double> %x) nounwind uwtable readnone ssp {
entry:
  %div.i = fdiv <4 x double> %x, %y
  ret <4 x double> %div.i
}

; CHECK: vdivpd LCP{{.*}}(%rip)
define <4 x double> @divpd256fold(<4 x double> %y) nounwind uwtable readnone ssp {
entry:
  %div.i = fdiv <4 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <4 x double> %div.i
}

; CHECK: vdivps
define <8 x float> @divps256(<8 x float> %y, <8 x float> %x) nounwind uwtable readnone ssp {
entry:
  %div.i = fdiv <8 x float> %x, %y
  ret <8 x float> %div.i
}

; CHECK: vdivps LCP{{.*}}(%rip)
define <8 x float> @divps256fold(<8 x float> %y) nounwind uwtable readnone ssp {
entry:
  %div.i = fdiv <8 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000>
  ret <8 x float> %div.i
}

