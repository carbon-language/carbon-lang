; RUN: llc -mtriple=x86_64-pc-linux -mcpu=corei7 < %s | FileCheck -check-prefix=CHECK -check-prefix=SSE2 %s
; RUN: llc -mtriple=x86_64-pc-linux -mattr=-sse4.1 -mcpu=corei7 < %s | FileCheck -check-prefix=CHECK -check-prefix=SSE2 %s
; RUN: llc -mtriple=x86_64-pc-linux -mcpu=corei7-avx < %s | FileCheck -check-prefix=CHECK -check-prefix=AVX %s

; Ensure that the backend selects SSE/AVX scalar fp instructions
; from a packed fp instrution plus a vector insert.


define <4 x float> @test_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fadd <4 x float> %a, %b
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test_add_ss
; SSE2: addss   %xmm1, %xmm0
; AVX: vaddss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fsub <4 x float> %a, %b
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test_sub_ss
; SSE2: subss   %xmm1, %xmm0
; AVX: vsubss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fmul <4 x float> %a, %b
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test_mul_ss
; SSE2: mulss   %xmm1, %xmm0
; AVX: vmulss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fdiv <4 x float> %a, %b
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test_div_ss
; SSE2: divss   %xmm1, %xmm0
; AVX: vdivss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fadd <2 x double> %a, %b
  %2 = shufflevector <2 x double> %1, <2 x double> %a, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test_add_sd
; SSE2: addsd   %xmm1, %xmm0
; AVX: vaddsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fsub <2 x double> %a, %b
  %2 = shufflevector <2 x double> %1, <2 x double> %a, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test_sub_sd
; SSE2: subsd   %xmm1, %xmm0
; AVX: vsubsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fmul <2 x double> %a, %b
  %2 = shufflevector <2 x double> %1, <2 x double> %a, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test_mul_sd
; SSE2: mulsd   %xmm1, %xmm0
; AVX: vmulsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fdiv <2 x double> %a, %b
  %2 = shufflevector <2 x double> %1, <2 x double> %a, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test_div_sd
; SSE2: divsd   %xmm1, %xmm0
; AVX: vdivsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <4 x float> @test2_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fadd <4 x float> %b, %a
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test2_add_ss
; SSE2: addss   %xmm0, %xmm1
; AVX: vaddss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fsub <4 x float> %b, %a
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test2_sub_ss
; SSE2: subss   %xmm0, %xmm1
; AVX: vsubss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fmul <4 x float> %b, %a
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test2_mul_ss
; SSE2: mulss   %xmm0, %xmm1
; AVX: vmulss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test2_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fdiv <4 x float> %b, %a
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}

; CHECK-LABEL: test2_div_ss
; SSE2: divss   %xmm0, %xmm1
; AVX: vdivss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test2_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fadd <2 x double> %b, %a
  %2 = shufflevector <2 x double> %1, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test2_add_sd
; SSE2: addsd   %xmm0, %xmm1
; AVX: vaddsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fsub <2 x double> %b, %a
  %2 = shufflevector <2 x double> %1, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test2_sub_sd
; SSE2: subsd   %xmm0, %xmm1
; AVX: vsubsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fmul <2 x double> %b, %a
  %2 = shufflevector <2 x double> %1, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test2_mul_sd
; SSE2: mulsd   %xmm0, %xmm1
; AVX: vmulsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test2_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fdiv <2 x double> %b, %a
  %2 = shufflevector <2 x double> %1, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %2
}

; CHECK-LABEL: test2_div_sd
; SSE2: divsd   %xmm0, %xmm1
; AVX: vdivsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <4 x float> @test3_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fadd <4 x float> %a, %b
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test3_add_ss
; SSE2: addss   %xmm1, %xmm0
; AVX: vaddss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test3_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fsub <4 x float> %a, %b
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test3_sub_ss
; SSE2: subss   %xmm1, %xmm0
; AVX: vsubss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test3_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fmul <4 x float> %a, %b
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test3_mul_ss
; SSE2: mulss   %xmm1, %xmm0
; AVX: vmulss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test3_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fdiv <4 x float> %a, %b
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test3_div_ss
; SSE2: divss   %xmm1, %xmm0
; AVX: vdivss   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test3_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fadd <2 x double> %a, %b
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %a, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test3_add_sd
; SSE2: addsd   %xmm1, %xmm0
; AVX: vaddsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test3_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fsub <2 x double> %a, %b
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %a, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test3_sub_sd
; SSE2: subsd   %xmm1, %xmm0
; AVX: vsubsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test3_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fmul <2 x double> %a, %b
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %a, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test3_mul_sd
; SSE2: mulsd   %xmm1, %xmm0
; AVX: vmulsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test3_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fdiv <2 x double> %a, %b
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %a, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test3_div_sd
; SSE2: divsd   %xmm1, %xmm0
; AVX: vdivsd   %xmm1, %xmm0, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <4 x float> @test4_add_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fadd <4 x float> %b, %a
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %b, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test4_add_ss
; SSE2: addss   %xmm0, %xmm1
; AVX: vaddss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test4_sub_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fsub <4 x float> %b, %a
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %b, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test4_sub_ss
; SSE2: subss   %xmm0, %xmm1
; AVX: vsubss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test4_mul_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fmul <4 x float> %b, %a
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %b, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test4_mul_ss
; SSE2: mulss   %xmm0, %xmm1
; AVX: vmulss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <4 x float> @test4_div_ss(<4 x float> %a, <4 x float> %b) {
  %1 = fdiv <4 x float> %b, %a
  %2 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %b, <4 x float> %1
  ret <4 x float> %2
}

; CHECK-LABEL: test4_div_ss
; SSE2: divss   %xmm0, %xmm1
; AVX: vdivss   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movss
; CHECK: ret


define <2 x double> @test4_add_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fadd <2 x double> %b, %a
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %b, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test4_add_sd
; SSE2: addsd   %xmm0, %xmm1
; AVX: vaddsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test4_sub_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fsub <2 x double> %b, %a
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %b, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test4_sub_sd
; SSE2: subsd   %xmm0, %xmm1
; AVX: vsubsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test4_mul_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fmul <2 x double> %b, %a
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %b, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test4_mul_sd
; SSE2: mulsd   %xmm0, %xmm1
; AVX: vmulsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret


define <2 x double> @test4_div_sd(<2 x double> %a, <2 x double> %b) {
  %1 = fdiv <2 x double> %b, %a
  %2 = select <2 x i1> <i1 false, i1 true>, <2 x double> %b, <2 x double> %1
  ret <2 x double> %2
}

; CHECK-LABEL: test4_div_sd
; SSE2: divsd   %xmm0, %xmm1
; AVX: vdivsd   %xmm0, %xmm1, %xmm0
; CHECK-NOT: movsd
; CHECK: ret

