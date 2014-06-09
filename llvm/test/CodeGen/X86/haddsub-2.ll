; RUN: llc < %s -march=x86-64 -mattr=+sse2,+sse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE3
; RUN: llc < %s -march=x86-64 -mattr=+sse2,+sse3,+ssse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSSE3
; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: llc < %s -march=x86-64 -mcpu=core-avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2



define <4 x float> @hadd_ps_test1(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 0
  %vecext1 = extractelement <4 x float> %A, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %A, i32 2
  %vecext3 = extractelement <4 x float> %A, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 1
  %vecext6 = extractelement <4 x float> %B, i32 0
  %vecext7 = extractelement <4 x float> %B, i32 1
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %add8, i32 2
  %vecext10 = extractelement <4 x float> %B, i32 2
  %vecext11 = extractelement <4 x float> %B, i32 3
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %add12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hadd_ps_test1
; CHECK: haddps
; CHECK-NEXT: ret


define <4 x float> @hadd_ps_test2(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 2
  %vecext1 = extractelement <4 x float> %A, i32 3
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 1
  %vecext2 = extractelement <4 x float> %A, i32 0
  %vecext3 = extractelement <4 x float> %A, i32 1
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 0
  %vecext6 = extractelement <4 x float> %B, i32 2
  %vecext7 = extractelement <4 x float> %B, i32 3
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %add8, i32 3
  %vecext10 = extractelement <4 x float> %B, i32 0
  %vecext11 = extractelement <4 x float> %B, i32 1
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %add12, i32 2
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hadd_ps_test2
; CHECK: haddps
; CHECK-NEXT: ret


define <4 x float> @hsub_ps_test1(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 0
  %vecext1 = extractelement <4 x float> %A, i32 1
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %sub, i32 0
  %vecext2 = extractelement <4 x float> %A, i32 2
  %vecext3 = extractelement <4 x float> %A, i32 3
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %sub4, i32 1
  %vecext6 = extractelement <4 x float> %B, i32 0
  %vecext7 = extractelement <4 x float> %B, i32 1
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %sub8, i32 2
  %vecext10 = extractelement <4 x float> %B, i32 2
  %vecext11 = extractelement <4 x float> %B, i32 3
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %sub12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hsub_ps_test1
; CHECK: hsubps
; CHECK-NEXT: ret


define <4 x float> @hsub_ps_test2(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 2
  %vecext1 = extractelement <4 x float> %A, i32 3
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %sub, i32 1
  %vecext2 = extractelement <4 x float> %A, i32 0
  %vecext3 = extractelement <4 x float> %A, i32 1
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %sub4, i32 0
  %vecext6 = extractelement <4 x float> %B, i32 3
  %vecext7 = extractelement <4 x float> %B, i32 2
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %sub8, i32 3
  %vecext10 = extractelement <4 x float> %B, i32 1
  %vecext11 = extractelement <4 x float> %B, i32 0
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %sub12, i32 2
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hsub_ps_test2
; CHECK: hsubps
; CHECK-NEXT: ret


define <4 x i32> @phadd_d_test1(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 0
  %vecext1 = extractelement <4 x i32> %A, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <4 x i32> %A, i32 2
  %vecext3 = extractelement <4 x i32> %A, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %add4, i32 1
  %vecext6 = extractelement <4 x i32> %B, i32 0
  %vecext7 = extractelement <4 x i32> %B, i32 1
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %add8, i32 2
  %vecext10 = extractelement <4 x i32> %B, i32 2
  %vecext11 = extractelement <4 x i32> %B, i32 3
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %add12, i32 3
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phadd_d_test1
; SSE3-NOT: phaddd
; SSSE3: phaddd
; AVX: vphaddd
; AVX2 vphaddd
; CHECK: ret


define <4 x i32> @phadd_d_test2(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 2
  %vecext1 = extractelement <4 x i32> %A, i32 3
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %add, i32 1
  %vecext2 = extractelement <4 x i32> %A, i32 0
  %vecext3 = extractelement <4 x i32> %A, i32 1
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %add4, i32 0
  %vecext6 = extractelement <4 x i32> %B, i32 2
  %vecext7 = extractelement <4 x i32> %B, i32 3
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %add8, i32 3
  %vecext10 = extractelement <4 x i32> %B, i32 0
  %vecext11 = extractelement <4 x i32> %B, i32 1
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %add12, i32 2
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phadd_d_test2
; SSE3-NOT: phaddd
; SSSE3: phaddd
; AVX: vphaddd
; AVX2 vphaddd
; CHECK: ret


define <4 x i32> @phsub_d_test1(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 0
  %vecext1 = extractelement <4 x i32> %A, i32 1
  %sub = sub i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %sub, i32 0
  %vecext2 = extractelement <4 x i32> %A, i32 2
  %vecext3 = extractelement <4 x i32> %A, i32 3
  %sub4 = sub i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %sub4, i32 1
  %vecext6 = extractelement <4 x i32> %B, i32 0
  %vecext7 = extractelement <4 x i32> %B, i32 1
  %sub8 = sub i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %sub8, i32 2
  %vecext10 = extractelement <4 x i32> %B, i32 2
  %vecext11 = extractelement <4 x i32> %B, i32 3
  %sub12 = sub i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %sub12, i32 3
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phsub_d_test1
; SSE3-NOT: phsubd
; SSSE3: phsubd
; AVX: vphsubd
; AVX2 vphsubd
; CHECK: ret


define <4 x i32> @phsub_d_test2(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 2
  %vecext1 = extractelement <4 x i32> %A, i32 3
  %sub = sub i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %sub, i32 1
  %vecext2 = extractelement <4 x i32> %A, i32 0
  %vecext3 = extractelement <4 x i32> %A, i32 1
  %sub4 = sub i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %sub4, i32 0
  %vecext6 = extractelement <4 x i32> %B, i32 3
  %vecext7 = extractelement <4 x i32> %B, i32 2
  %sub8 = sub i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %sub8, i32 3
  %vecext10 = extractelement <4 x i32> %B, i32 1
  %vecext11 = extractelement <4 x i32> %B, i32 0
  %sub12 = sub i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %sub12, i32 2
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phsub_d_test2
; SSE3-NOT: phsubd
; SSSE3: phsubd
; AVX: vphsubd
; AVX2 vphsubd
; CHECK: ret


define <2 x double> @hadd_pd_test1(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 0
  %vecext1 = extractelement <2 x double> %A, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %add, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 0
  %vecext3 = extractelement <2 x double> %B, i32 1
  %add2 = fadd double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %add2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hadd_pd_test1
; CHECK: haddpd
; CHECK-NEXT: ret


define <2 x double> @hadd_pd_test2(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 1
  %vecext1 = extractelement <2 x double> %A, i32 0
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %add, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 1
  %vecext3 = extractelement <2 x double> %B, i32 0
  %add2 = fadd double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %add2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hadd_pd_test2
; CHECK: haddpd
; CHECK-NEXT: ret


define <2 x double> @hsub_pd_test1(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 0
  %vecext1 = extractelement <2 x double> %A, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 0
  %vecext3 = extractelement <2 x double> %B, i32 1
  %sub2 = fsub double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %sub2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hsub_pd_test1
; CHECK: hsubpd
; CHECK-NEXT: ret


define <2 x double> @hsub_pd_test2(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 1
  %vecext1 = extractelement <2 x double> %A, i32 0
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 1
  %vecext3 = extractelement <2 x double> %B, i32 0
  %sub2 = fsub double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %sub2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hsub_pd_test2
; CHECK: hsubpd
; CHECK-NEXT: ret


define <4 x double> @avx_vhadd_pd_test(<4 x double> %A, <4 x double> %B) {
  %vecext = extractelement <4 x double> %A, i32 0
  %vecext1 = extractelement <4 x double> %A, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %add, i32 0
  %vecext2 = extractelement <4 x double> %A, i32 2
  %vecext3 = extractelement <4 x double> %A, i32 3
  %add4 = fadd double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %add4, i32 1
  %vecext6 = extractelement <4 x double> %B, i32 0
  %vecext7 = extractelement <4 x double> %B, i32 1
  %add8 = fadd double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %add8, i32 2
  %vecext10 = extractelement <4 x double> %B, i32 2
  %vecext11 = extractelement <4 x double> %B, i32 3
  %add12 = fadd double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %add12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_vhadd_pd_test
; SSE3: haddpd
; SSE3-NEXT: haddpd
; SSSE3: haddpd
; SSSE3: haddpd
; AVX: vhaddpd
; AVX2: vhaddpd
; CHECK: ret


define <4 x double> @avx_vhsub_pd_test(<4 x double> %A, <4 x double> %B) {
  %vecext = extractelement <4 x double> %A, i32 0
  %vecext1 = extractelement <4 x double> %A, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <4 x double> %A, i32 2
  %vecext3 = extractelement <4 x double> %A, i32 3
  %sub4 = fsub double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %sub4, i32 1
  %vecext6 = extractelement <4 x double> %B, i32 0
  %vecext7 = extractelement <4 x double> %B, i32 1
  %sub8 = fsub double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %sub8, i32 2
  %vecext10 = extractelement <4 x double> %B, i32 2
  %vecext11 = extractelement <4 x double> %B, i32 3
  %sub12 = fsub double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %sub12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_vhsub_pd_test
; SSE3: hsubpd
; SSE3-NEXT: hsubpd
; SSSE3: hsubpd
; SSSE3-NEXT: hsubpd
; AVX: vhsubpd
; AVX2: vhsubpd
; CHECK: ret


define <8 x i32> @avx2_vphadd_d_test(<8 x i32> %A, <8 x i32> %B) {
  %vecext = extractelement <8 x i32> %A, i32 0
  %vecext1 = extractelement <8 x i32> %A, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %A, i32 2
  %vecext3 = extractelement <8 x i32> %A, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 1
  %vecext6 = extractelement <8 x i32> %A, i32 4
  %vecext7 = extractelement <8 x i32> %A, i32 5
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <8 x i32> %vecinit5, i32 %add8, i32 2
  %vecext10 = extractelement <8 x i32> %A, i32 6
  %vecext11 = extractelement <8 x i32> %A, i32 7
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <8 x i32> %vecinit9, i32 %add12, i32 3
  %vecext14 = extractelement <8 x i32> %B, i32 0
  %vecext15 = extractelement <8 x i32> %B, i32 1
  %add16 = add i32 %vecext14, %vecext15
  %vecinit17 = insertelement <8 x i32> %vecinit13, i32 %add16, i32 4
  %vecext18 = extractelement <8 x i32> %B, i32 2
  %vecext19 = extractelement <8 x i32> %B, i32 3
  %add20 = add i32 %vecext18, %vecext19
  %vecinit21 = insertelement <8 x i32> %vecinit17, i32 %add20, i32 5
  %vecext22 = extractelement <8 x i32> %B, i32 4
  %vecext23 = extractelement <8 x i32> %B, i32 5
  %add24 = add i32 %vecext22, %vecext23
  %vecinit25 = insertelement <8 x i32> %vecinit21, i32 %add24, i32 6
  %vecext26 = extractelement <8 x i32> %B, i32 6
  %vecext27 = extractelement <8 x i32> %B, i32 7
  %add28 = add i32 %vecext26, %vecext27
  %vecinit29 = insertelement <8 x i32> %vecinit25, i32 %add28, i32 7
  ret <8 x i32> %vecinit29
}
; CHECK-LABEL: avx2_vphadd_d_test
; SSE3-NOT: phaddd
; SSSE3: phaddd
; SSSE3-NEXT: phaddd
; AVX-NOT: vphaddd
; AVX2: vphaddd
; CHECK: ret

