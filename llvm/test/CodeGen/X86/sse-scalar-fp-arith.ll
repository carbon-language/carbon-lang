; RUN: llc -mtriple=x86_64-pc-linux -mcpu=corei7 < %s | FileCheck -check-prefix=CHECK -check-prefix=SSE2 %s
; RUN: llc -mtriple=x86_64-pc-linux -mattr=-sse4.1 -mcpu=corei7 < %s | FileCheck -check-prefix=CHECK -check-prefix=SSE2 %s
; RUN: llc -mtriple=x86_64-pc-linux -mcpu=corei7-avx < %s | FileCheck -check-prefix=CHECK -check-prefix=AVX %s

; Ensure that the backend no longer emits unnecessary vector insert
; instructions immediately after SSE scalar fp instructions
; like addss or mulss.


define <4 x float> @test_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %add = fadd float %2, %1
  %3 = insertelement <4 x float> %a, float %add, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_add_ss
; SSE2: addss   %xmm1, %xmm0
; AVX: vaddss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %sub = fsub float %2, %1
  %3 = insertelement <4 x float> %a, float %sub, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_sub_ss
; SSE2: subss   %xmm1, %xmm0
; AVX: vsubss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret

define <4 x float> @test_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %mul = fmul float %2, %1
  %3 = insertelement <4 x float> %a, float %mul, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_mul_ss
; SSE2: mulss   %xmm1, %xmm0
; AVX: vmulss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %div = fdiv float %2, %1
  %3 = insertelement <4 x float> %a, float %div, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_div_ss
; SSE2: divss   %xmm1, %xmm0
; AVX: vdivss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %b, i32 0
  %2 = extractelement <2 x double> %a, i32 0
  %add = fadd double %2, %1
  %3 = insertelement <2 x double> %a, double %add, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test_add_sd
; SSE2: addsd   %xmm1, %xmm0
; AVX: vaddsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %b, i32 0
  %2 = extractelement <2 x double> %a, i32 0
  %sub = fsub double %2, %1
  %3 = insertelement <2 x double> %a, double %sub, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test_sub_sd
; SSE2: subsd   %xmm1, %xmm0
; AVX: vsubsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %b, i32 0
  %2 = extractelement <2 x double> %a, i32 0
  %mul = fmul double %2, %1
  %3 = insertelement <2 x double> %a, double %mul, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test_mul_sd
; SSE2: mulsd   %xmm1, %xmm0
; AVX: vmulsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %b, i32 0
  %2 = extractelement <2 x double> %a, i32 0
  %div = fdiv double %2, %1
  %3 = insertelement <2 x double> %a, double %div, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test_div_sd
; SSE2: divsd   %xmm1, %xmm0
; AVX: vdivsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <4 x float> @test2_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %a, i32 0
  %2 = extractelement <4 x float> %b, i32 0
  %add = fadd float %1, %2
  %3 = insertelement <4 x float> %b, float %add, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test2_add_ss
; SSE2: addss   %xmm0, %xmm1
; AVX: vaddss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %a, i32 0
  %2 = extractelement <4 x float> %b, i32 0
  %sub = fsub float %2, %1
  %3 = insertelement <4 x float> %b, float %sub, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test2_sub_ss
; SSE2: subss   %xmm0, %xmm1
; AVX: vsubss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %a, i32 0
  %2 = extractelement <4 x float> %b, i32 0
  %mul = fmul float %1, %2
  %3 = insertelement <4 x float> %b, float %mul, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test2_mul_ss
; SSE2: mulss   %xmm0, %xmm1
; AVX: vmulss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %a, i32 0
  %2 = extractelement <4 x float> %b, i32 0
  %div = fdiv float %2, %1
  %3 = insertelement <4 x float> %b, float %div, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test2_div_ss
; SSE2: divss   %xmm0, %xmm1
; AVX: vdivss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test2_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %a, i32 0
  %2 = extractelement <2 x double> %b, i32 0
  %add = fadd double %1, %2
  %3 = insertelement <2 x double> %b, double %add, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test2_add_sd
; SSE2: addsd   %xmm0, %xmm1
; AVX: vaddsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %a, i32 0
  %2 = extractelement <2 x double> %b, i32 0
  %sub = fsub double %2, %1
  %3 = insertelement <2 x double> %b, double %sub, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test2_sub_sd
; SSE2: subsd   %xmm0, %xmm1
; AVX: vsubsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %a, i32 0
  %2 = extractelement <2 x double> %b, i32 0
  %mul = fmul double %1, %2
  %3 = insertelement <2 x double> %b, double %mul, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test2_mul_sd
; SSE2: mulsd   %xmm0, %xmm1
; AVX: vmulsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = extractelement <2 x double> %a, i32 0
  %2 = extractelement <2 x double> %b, i32 0
  %div = fdiv double %2, %1
  %3 = insertelement <2 x double> %b, double %div, i32 0
  ret <2 x double> %3
}

; CHECK-LABEL: test2_div_sd
; SSE2: divsd   %xmm0, %xmm1
; AVX: vdivsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <4 x float> @test_multiple_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %add = fadd float %2, %1
  %add2 = fadd float %2, %add
  %3 = insertelement <4 x float> %a, float %add2, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_multiple_add_ss
; CHECK: addss
; CHECK: addss
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_multiple_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %sub = fsub float %2, %1
  %sub2 = fsub float %2, %sub
  %3 = insertelement <4 x float> %a, float %sub2, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_multiple_sub_ss
; CHECK: subss
; CHECK: subss
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_multiple_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %mul = fmul float %2, %1
  %mul2 = fmul float %2, %mul
  %3 = insertelement <4 x float> %a, float %mul2, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_multiple_mul_ss
; CHECK: mulss
; CHECK: mulss
; CHECK-NOT: movss
; CHECK: ret

define <4 x float> @test_multiple_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = extractelement <4 x float> %b, i32 0
  %2 = extractelement <4 x float> %a, i32 0
  %div = fdiv float %2, %1
  %div2 = fdiv float %2, %div
  %3 = insertelement <4 x float> %a, float %div2, i32 0
  ret <4 x float> %3
}

; CHECK-LABEL: test_multiple_div_ss
; CHECK: divss
; CHECK: divss
; CHECK-NOT: movss
; CHECK: ret

