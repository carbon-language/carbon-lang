; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL  --check-prefix=AVX

;
; Double to Signed Integer
;

define <2 x i64> @fptosi_2vf64(<2 x double> %a) {
; SSE2-LABEL: fptosi_2vf64:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_2vf64:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm1[0],xmm0[0]
; AVX-NEXT:    retq
  %cvt = fptosi <2 x double> %a to <2 x i64>
  ret <2 x i64> %cvt
}

define <4 x i32> @fptosi_2vf64_i32(<2 x double> %a) {
; SSE2-LABEL: fptosi_2vf64_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,2,2,3]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_2vf64_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm1[0],xmm0[0]
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX-NEXT:    retq
  %cvt = fptosi <2 x double> %a to <2 x i32>
  %ext = shufflevector <2 x i32> %cvt, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x i32> %ext
}

define <4 x i64> @fptosi_4vf64(<4 x double> %a) {
; SSE2-LABEL: fptosi_4vf64:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm0[0]
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm3
; SSE2-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm3 = xmm3[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    movdqa %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vcvttsd2si %xmm1, %rax
; AVX-NEXT:    vmovq %rax, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm1 = xmm1[1,0]
; AVX-NEXT:    vcvttsd2si %xmm1, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm2[0],xmm1[0]
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %cvt = fptosi <4 x double> %a to <4 x i64>
  ret <4 x i64> %cvt
}

define <4 x i32> @fptosi_4vf64_i32(<4 x double> %a) {
; SSE2-LABEL: fptosi_4vf64_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[0,2,2,3]
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm0[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,2,3]
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf64_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttpd2dqy %ymm0, %xmm0
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %cvt = fptosi <4 x double> %a to <4 x i32>
  ret <4 x i32> %cvt
}

;
; Double to Unsigned Integer
;

define <2 x i64> @fptoui_2vf64(<2 x double> %a) {
; SSE2-LABEL: fptoui_2vf64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movsd {{.*#+}} xmm2 = mem[0],zero
; SSE2-NEXT:    movapd %xmm0, %xmm1
; SSE2-NEXT:    subsd %xmm2, %xmm1
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttsd2si %xmm0, %rdx
; SSE2-NEXT:    ucomisd %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rdx
; SSE2-NEXT:    movd %rdx, %xmm1
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    movapd %xmm0, %xmm3
; SSE2-NEXT:    subsd %xmm2, %xmm3
; SSE2-NEXT:    cvttsd2si %xmm3, %rax
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttsd2si %xmm0, %rcx
; SSE2-NEXT:    ucomisd %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rcx
; SSE2-NEXT:    movd %rcx, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_2vf64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovsd {{.*#+}} xmm1 = mem[0],zero
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vcvttsd2si %xmm2, %rax
; AVX-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rdx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm3
; AVX-NEXT:    vcvttsd2si %xmm3, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rcx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rcx
; AVX-NEXT:    vmovq %rcx, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    retq
  %cvt = fptoui <2 x double> %a to <2 x i64>
  ret <2 x i64> %cvt
}

define <4 x i32> @fptoui_2vf64_i32(<2 x double> %a) {
; SSE2-LABEL: fptoui_2vf64_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    movapd %xmm0, %xmm2
; SSE2-NEXT:    subsd %xmm1, %xmm2
; SSE2-NEXT:    cvttsd2si %xmm2, %rax
; SSE2-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttsd2si %xmm0, %rdx
; SSE2-NEXT:    ucomisd %xmm1, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rdx
; SSE2-NEXT:    movd %rdx, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    movapd %xmm0, %xmm3
; SSE2-NEXT:    subsd %xmm1, %xmm3
; SSE2-NEXT:    cvttsd2si %xmm3, %rax
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttsd2si %xmm0, %rcx
; SSE2-NEXT:    ucomisd %xmm1, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rcx
; SSE2-NEXT:    movd %rcx, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm0[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,2,3]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_2vf64_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovsd {{.*#+}} xmm1 = mem[0],zero
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vcvttsd2si %xmm2, %rax
; AVX-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rdx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm3
; AVX-NEXT:    vcvttsd2si %xmm3, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rcx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rcx
; AVX-NEXT:    vmovq %rcx, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX-NEXT:    retq
  %cvt = fptoui <2 x double> %a to <2 x i32>
  %ext = shufflevector <2 x i32> %cvt, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x i32> %ext
}

define <4 x i64> @fptoui_4vf64(<4 x double> %a) {
; SSE2-LABEL: fptoui_4vf64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movapd %xmm0, %xmm2
; SSE2-NEXT:    movsd {{.*#+}} xmm3 = mem[0],zero
; SSE2-NEXT:    {{.*#+}} kill: XMM0<def> XMM2<kill>
; SSE2-NEXT:    subsd %xmm3, %xmm0
; SSE2-NEXT:    cvttsd2si %xmm0, %rcx
; SSE2-NEXT:    movabsq $-9223372036854775808, %rax # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm2, %rdx
; SSE2-NEXT:    ucomisd %xmm3, %xmm2
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm0
; SSE2-NEXT:    shufpd {{.*#+}} xmm2 = xmm2[1,0]
; SSE2-NEXT:    movapd %xmm2, %xmm4
; SSE2-NEXT:    subsd %xmm3, %xmm4
; SSE2-NEXT:    cvttsd2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm2, %rdx
; SSE2-NEXT:    ucomisd %xmm3, %xmm2
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm2
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE2-NEXT:    movapd %xmm1, %xmm2
; SSE2-NEXT:    subsd %xmm3, %xmm2
; SSE2-NEXT:    cvttsd2si %xmm2, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm1, %rdx
; SSE2-NEXT:    ucomisd %xmm3, %xmm1
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE2-NEXT:    movapd %xmm1, %xmm4
; SSE2-NEXT:    subsd %xmm3, %xmm4
; SSE2-NEXT:    cvttsd2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm1, %rax
; SSE2-NEXT:    ucomisd %xmm3, %xmm1
; SSE2-NEXT:    cmovaeq %rcx, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vmovsd {{.*#+}} xmm1 = mem[0],zero
; AVX-NEXT:    vsubsd %xmm1, %xmm2, %xmm3
; AVX-NEXT:    vcvttsd2si %xmm3, %rax
; AVX-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm2, %rdx
; AVX-NEXT:    vucomisd %xmm1, %xmm2
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm3
; AVX-NEXT:    vpermilpd {{.*#+}} xmm2 = xmm2[1,0]
; AVX-NEXT:    vsubsd %xmm1, %xmm2, %xmm4
; AVX-NEXT:    vcvttsd2si %xmm4, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm2, %rdx
; AVX-NEXT:    vucomisd %xmm1, %xmm2
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm2
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm3[0],xmm2[0]
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm3
; AVX-NEXT:    vcvttsd2si %xmm3, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rdx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm3
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vsubsd %xmm1, %xmm0, %xmm4
; AVX-NEXT:    vcvttsd2si %xmm4, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rcx
; AVX-NEXT:    vucomisd %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rcx
; AVX-NEXT:    vmovq %rcx, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm3[0],xmm0[0]
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    retq
  %cvt = fptoui <4 x double> %a to <4 x i64>
  ret <4 x i64> %cvt
}

define <4 x i32> @fptoui_4vf64_i32(<4 x double> %a) {
; SSE2-LABEL: fptoui_4vf64_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movsd {{.*#+}} xmm2 = mem[0],zero
; SSE2-NEXT:    movapd %xmm1, %xmm3
; SSE2-NEXT:    subsd %xmm2, %xmm3
; SSE2-NEXT:    cvttsd2si %xmm3, %rcx
; SSE2-NEXT:    movabsq $-9223372036854775808, %rax # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm1, %rdx
; SSE2-NEXT:    ucomisd %xmm2, %xmm1
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm3
; SSE2-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE2-NEXT:    movapd %xmm1, %xmm4
; SSE2-NEXT:    subsd %xmm2, %xmm4
; SSE2-NEXT:    cvttsd2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm1, %rdx
; SSE2-NEXT:    ucomisd %xmm2, %xmm1
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm3 = xmm3[0],xmm1[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm3[0,2,2,3]
; SSE2-NEXT:    movapd %xmm0, %xmm3
; SSE2-NEXT:    subsd %xmm2, %xmm3
; SSE2-NEXT:    cvttsd2si %xmm3, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm0, %rdx
; SSE2-NEXT:    ucomisd %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm3
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    movapd %xmm0, %xmm4
; SSE2-NEXT:    subsd %xmm2, %xmm4
; SSE2-NEXT:    cvttsd2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttsd2si %xmm0, %rax
; SSE2-NEXT:    ucomisd %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rcx, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm3 = xmm3[0],xmm0[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm3[0,2,2,3]
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf64_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpermilpd {{.*#+}} xmm1 = xmm0[1,0]
; AVX-NEXT:    vcvttsd2si %xmm1, %rax
; AVX-NEXT:    vcvttsd2si %xmm0, %rcx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrd $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; AVX-NEXT:    vcvttsd2si %xmm0, %rax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %cvt = fptoui <4 x double> %a to <4 x i32>
  ret <4 x i32> %cvt
}

;
; Float to Signed Integer
;

define <4 x i32> @fptosi_4vf32(<4 x float> %a) {
; SSE2-LABEL: fptosi_4vf32:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttps2dq %xmm0, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttps2dq %xmm0, %xmm0
; AVX-NEXT:    retq
  %cvt = fptosi <4 x float> %a to <4 x i32>
  ret <4 x i32> %cvt
}

define <2 x i64> @fptosi_4vf32_i64(<4 x float> %a) {
; SSE2-LABEL: fptosi_4vf32_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vmovshdup {{.*#+}} xmm0 = xmm0[1,1,3,3]
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm1[0],xmm0[0]
; AVX-NEXT:    retq
  %shuf = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 0, i32 1>
  %cvt = fptosi <2 x float> %shuf to <2 x i64>
  ret <2 x i64> %cvt
}

define <8 x i32> @fptosi_8vf32(<8 x float> %a) {
; SSE2-LABEL: fptosi_8vf32:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttps2dq %xmm0, %xmm0
; SSE2-NEXT:    cvttps2dq %xmm1, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_8vf32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvttps2dq %ymm0, %ymm0
; AVX-NEXT:    retq
  %cvt = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %cvt
}

define <4 x i64> @fptosi_8vf32_i64(<8 x float> %a) {
; SSE2-LABEL: fptosi_8vf32_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    movaps %xmm0, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    movaps %xmm0, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movd %rax, %xmm3
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm3[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_8vf32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpermilps {{.*#+}} xmm1 = xmm0[3,1,2,3]
; AVX-NEXT:    vcvttss2si %xmm1, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vpermilpd {{.*#+}} xmm2 = xmm0[1,0]
; AVX-NEXT:    vcvttss2si %xmm2, %rax
; AVX-NEXT:    vmovq %rax, %xmm2
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm2[0],xmm1[0]
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm2
; AVX-NEXT:    vmovshdup {{.*#+}} xmm0 = xmm0[1,1,3,3]
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x float> %a, <8 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = fptosi <4 x float> %shuf to <4 x i64>
  ret <4 x i64> %cvt
}

;
; Float to Unsigned Integer
;

define <4 x i32> @fptoui_4vf32(<4 x float> %a) {
; SSE2-LABEL: fptoui_4vf32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps %xmm0, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    movaps %xmm0, %xmm2
; SSE2-NEXT:    shufps {{.*#+}} xmm2 = xmm2[1,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm2, %rax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovshdup {{.*#+}} xmm1 = xmm0[1,1,3,3]
; AVX-NEXT:    vcvttss2si %xmm1, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rcx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrd $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpermilpd {{.*#+}} xmm2 = xmm0[1,0]
; AVX-NEXT:    vcvttss2si %xmm2, %rax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %cvt = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %cvt
}

define <2 x i64> @fptoui_4vf32_i64(<4 x float> %a) {
; SSE2-LABEL: fptoui_4vf32_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movss {{.*#+}} xmm2 = mem[0],zero,zero,zero
; SSE2-NEXT:    movaps %xmm0, %xmm1
; SSE2-NEXT:    subss %xmm2, %xmm1
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttss2si %xmm0, %rdx
; SSE2-NEXT:    ucomiss %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rdx
; SSE2-NEXT:    movd %rdx, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE2-NEXT:    movaps %xmm0, %xmm3
; SSE2-NEXT:    subss %xmm2, %xmm3
; SSE2-NEXT:    cvttss2si %xmm3, %rax
; SSE2-NEXT:    xorq %rcx, %rax
; SSE2-NEXT:    cvttss2si %xmm0, %rcx
; SSE2-NEXT:    ucomiss %xmm2, %xmm0
; SSE2-NEXT:    cmovaeq %rax, %rcx
; SSE2-NEXT:    movd %rcx, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; AVX-NEXT:    vsubss %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vcvttss2si %xmm2, %rax
; AVX-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rdx
; AVX-NEXT:    vucomiss %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm2
; AVX-NEXT:    vmovshdup {{.*#+}} xmm0 = xmm0[1,1,3,3]
; AVX-NEXT:    vsubss %xmm1, %xmm0, %xmm3
; AVX-NEXT:    vcvttss2si %xmm3, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rcx
; AVX-NEXT:    vucomiss %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rcx
; AVX-NEXT:    vmovq %rcx, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    retq
  %shuf = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 0, i32 1>
  %cvt = fptoui <2 x float> %shuf to <2 x i64>
  ret <2 x i64> %cvt
}

define <8 x i32> @fptoui_8vf32(<8 x float> %a) {
; SSE2-LABEL: fptoui_8vf32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps %xmm0, %xmm2
; SSE2-NEXT:    {{.*#+}} kill: XMM0<def> XMM2<kill>
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    movaps %xmm2, %xmm3
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[1,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm3, %rax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1]
; SSE2-NEXT:    cvttss2si %xmm2, %rax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    shufpd {{.*#+}} xmm2 = xmm2[1,0]
; SSE2-NEXT:    cvttss2si %xmm2, %rax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1]
; SSE2-NEXT:    movaps %xmm1, %xmm2
; SSE2-NEXT:    shufps {{.*#+}} xmm2 = xmm2[3,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm2, %rax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    movaps %xmm1, %xmm3
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[1,1,2,3]
; SSE2-NEXT:    cvttss2si %xmm3, %rax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1]
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE2-NEXT:    cvttss2si %xmm1, %rax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE2-NEXT:    movdqa %xmm2, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_8vf32:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vmovshdup {{.*#+}} xmm2 = xmm1[1,1,3,3]
; AVX-NEXT:    vcvttss2si %xmm2, %rax
; AVX-NEXT:    vcvttss2si %xmm1, %rcx
; AVX-NEXT:    vmovd %ecx, %xmm2
; AVX-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm3 = xmm1[1,0]
; AVX-NEXT:    vcvttss2si %xmm3, %rax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-NEXT:    vpermilps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; AVX-NEXT:    vcvttss2si %xmm1, %rax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX-NEXT:    vmovshdup {{.*#+}} xmm2 = xmm0[1,1,3,3]
; AVX-NEXT:    vcvttss2si %xmm2, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rcx
; AVX-NEXT:    vmovd %ecx, %xmm2
; AVX-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm3 = xmm0[1,0]
; AVX-NEXT:    vcvttss2si %xmm3, %rax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX-NEXT:    vcvttss2si %xmm0, %rax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %cvt = fptoui <8 x float> %a to <8 x i32>
  ret <8 x i32> %cvt
}

define <4 x i64> @fptoui_8vf32_i64(<8 x float> %a) {
; SSE2-LABEL: fptoui_8vf32_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; SSE2-NEXT:    movaps %xmm0, %xmm2
; SSE2-NEXT:    subss %xmm1, %xmm2
; SSE2-NEXT:    cvttss2si %xmm2, %rcx
; SSE2-NEXT:    movabsq $-9223372036854775808, %rax # imm = 0x8000000000000000
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttss2si %xmm0, %rdx
; SSE2-NEXT:    ucomiss %xmm1, %xmm0
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm2
; SSE2-NEXT:    movaps %xmm0, %xmm3
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[1,1,2,3]
; SSE2-NEXT:    movaps %xmm3, %xmm4
; SSE2-NEXT:    subss %xmm1, %xmm4
; SSE2-NEXT:    cvttss2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttss2si %xmm3, %rdx
; SSE2-NEXT:    ucomiss %xmm1, %xmm3
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm3
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; SSE2-NEXT:    movaps %xmm0, %xmm3
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[3,1,2,3]
; SSE2-NEXT:    movaps %xmm3, %xmm4
; SSE2-NEXT:    subss %xmm1, %xmm4
; SSE2-NEXT:    cvttss2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttss2si %xmm3, %rdx
; SSE2-NEXT:    ucomiss %xmm1, %xmm3
; SSE2-NEXT:    cmovaeq %rcx, %rdx
; SSE2-NEXT:    movd %rdx, %xmm3
; SSE2-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE2-NEXT:    movapd %xmm0, %xmm4
; SSE2-NEXT:    subss %xmm1, %xmm4
; SSE2-NEXT:    cvttss2si %xmm4, %rcx
; SSE2-NEXT:    xorq %rax, %rcx
; SSE2-NEXT:    cvttss2si %xmm0, %rax
; SSE2-NEXT:    ucomiss %xmm1, %xmm0
; SSE2-NEXT:    cmovaeq %rcx, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm3[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_8vf32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpermilps {{.*#+}} xmm2 = xmm0[3,1,2,3]
; AVX-NEXT:    vmovss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; AVX-NEXT:    vsubss %xmm1, %xmm2, %xmm3
; AVX-NEXT:    vcvttss2si %xmm3, %rax
; AVX-NEXT:    movabsq $-9223372036854775808, %rcx # imm = 0x8000000000000000
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm2, %rdx
; AVX-NEXT:    vucomiss %xmm1, %xmm2
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm2
; AVX-NEXT:    vpermilpd {{.*#+}} xmm3 = xmm0[1,0]
; AVX-NEXT:    vsubss %xmm1, %xmm3, %xmm4
; AVX-NEXT:    vcvttss2si %xmm4, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm3, %rdx
; AVX-NEXT:    vucomiss %xmm1, %xmm3
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm3
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm3[0],xmm2[0]
; AVX-NEXT:    vsubss %xmm1, %xmm0, %xmm3
; AVX-NEXT:    vcvttss2si %xmm3, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rdx
; AVX-NEXT:    vucomiss %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rdx
; AVX-NEXT:    vmovq %rdx, %xmm3
; AVX-NEXT:    vmovshdup {{.*#+}} xmm0 = xmm0[1,1,3,3]
; AVX-NEXT:    vsubss %xmm1, %xmm0, %xmm4
; AVX-NEXT:    vcvttss2si %xmm4, %rax
; AVX-NEXT:    xorq %rcx, %rax
; AVX-NEXT:    vcvttss2si %xmm0, %rcx
; AVX-NEXT:    vucomiss %xmm1, %xmm0
; AVX-NEXT:    cmovaeq %rax, %rcx
; AVX-NEXT:    vmovq %rcx, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm3[0],xmm0[0]
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x float> %a, <8 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = fptoui <4 x float> %shuf to <4 x i64>
  ret <4 x i64> %cvt
}

;
; Constant Folding
;

define <2 x i64> @fptosi_2vf64c() {
; SSE2-LABEL: fptosi_2vf64c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,18446744073709551615]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_2vf64c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [1,18446744073709551615]
; AVX-NEXT:    retq
  %cvt = fptosi <2 x double> <double 1.0, double -1.0> to <2 x i64>
  ret <2 x i64> %cvt
}

define <4 x i32> @fptosi_2vf64c_i32() {
; SSE2-LABEL: fptosi_2vf64c_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = <4294967295,1,u,u>
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_2vf64c_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = <4294967295,1,u,u>
; AVX-NEXT:    retq
  %cvt = fptosi <2 x double> <double -1.0, double 1.0> to <2 x i32>
  %ext = shufflevector <2 x i32> %cvt, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x i32> %ext
}

define <4 x i64> @fptosi_4vf64c() {
; SSE2-LABEL: fptosi_4vf64c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,18446744073709551615]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [2,18446744073709551613]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf64c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [1,18446744073709551615,2,18446744073709551613]
; AVX-NEXT:    retq
  %cvt = fptosi <4 x double> <double 1.0, double -1.0, double 2.0, double -3.0> to <4 x i64>
  ret <4 x i64> %cvt
}

define <4 x i32> @fptosi_4vf64c_i32() {
; SSE2-LABEL: fptosi_4vf64c_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [4294967295,1,4294967294,3]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf64c_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [4294967295,1,4294967294,3]
; AVX-NEXT:    retq
  %cvt = fptosi <4 x double> <double -1.0, double 1.0, double -2.0, double 3.0> to <4 x i32>
  ret <4 x i32> %cvt
}

define <2 x i64> @fptoui_2vf64c() {
; SSE2-LABEL: fptoui_2vf64c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [2,4]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_2vf64c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [2,4]
; AVX-NEXT:    retq
  %cvt = fptoui <2 x double> <double 2.0, double 4.0> to <2 x i64>
  ret <2 x i64> %cvt
}

define <4 x i32> @fptoui_2vf64c_i32(<2 x double> %a) {
; SSE2-LABEL: fptoui_2vf64c_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = <2,4,u,u>
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_2vf64c_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = <2,4,u,u>
; AVX-NEXT:    retq
  %cvt = fptoui <2 x double> <double 2.0, double 4.0> to <2 x i32>
  %ext = shufflevector <2 x i32> %cvt, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x i32> %ext
}

define <4 x i64> @fptoui_4vf64c(<4 x double> %a) {
; SSE2-LABEL: fptoui_4vf64c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [2,4]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [6,8]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf64c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [2,4,6,8]
; AVX-NEXT:    retq
  %cvt = fptoui <4 x double> <double 2.0, double 4.0, double 6.0, double 8.0> to <4 x i64>
  ret <4 x i64> %cvt
}

define <4 x i32> @fptoui_4vf64c_i32(<4 x double> %a) {
; SSE2-LABEL: fptoui_4vf64c_i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [2,4,6,8]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf64c_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [2,4,6,8]
; AVX-NEXT:    retq
  %cvt = fptoui <4 x double> <double 2.0, double 4.0, double 6.0, double 8.0> to <4 x i32>
  ret <4 x i32> %cvt
}

define <4 x i32> @fptosi_4vf32c() {
; SSE2-LABEL: fptosi_4vf32c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,4294967295,2,3]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf32c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [1,4294967295,2,3]
; AVX-NEXT:    retq
  %cvt = fptosi <4 x float> <float 1.0, float -1.0, float 2.0, float 3.0> to <4 x i32>
  ret <4 x i32> %cvt
}

define <4 x i64> @fptosi_4vf32c_i64() {
; SSE2-LABEL: fptosi_4vf32c_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,18446744073709551615]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [2,3]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_4vf32c_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [1,18446744073709551615,2,3]
; AVX-NEXT:    retq
  %cvt = fptosi <4 x float> <float 1.0, float -1.0, float 2.0, float 3.0> to <4 x i64>
  ret <4 x i64> %cvt
}

define <8 x i32> @fptosi_8vf32c(<8 x float> %a) {
; SSE2-LABEL: fptosi_8vf32c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,4294967295,2,3]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [6,4294967288,2,4294967295]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptosi_8vf32c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [1,4294967295,2,3,6,4294967288,2,4294967295]
; AVX-NEXT:    retq
  %cvt = fptosi <8 x float> <float 1.0, float -1.0, float 2.0, float 3.0, float 6.0, float -8.0, float 2.0, float -1.0> to <8 x i32>
  ret <8 x i32> %cvt
}

define <4 x i32> @fptoui_4vf32c(<4 x float> %a) {
; SSE2-LABEL: fptoui_4vf32c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,2,4,6]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf32c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [1,2,4,6]
; AVX-NEXT:    retq
  %cvt = fptoui <4 x float> <float 1.0, float 2.0, float 4.0, float 6.0> to <4 x i32>
  ret <4 x i32> %cvt
}

define <4 x i64> @fptoui_4vf32c_i64() {
; SSE2-LABEL: fptoui_4vf32c_i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,2]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [4,8]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_4vf32c_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [1,2,4,8]
; AVX-NEXT:    retq
  %cvt = fptoui <4 x float> <float 1.0, float 2.0, float 4.0, float 8.0> to <4 x i64>
  ret <4 x i64> %cvt
}

define <8 x i32> @fptoui_8vf32c(<8 x float> %a) {
; SSE2-LABEL: fptoui_8vf32c:
; SSE2:       # BB#0:
; SSE2-NEXT:    movaps {{.*#+}} xmm0 = [1,2,4,6]
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [8,6,4,1]
; SSE2-NEXT:    retq
;
; AVX-LABEL: fptoui_8vf32c:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [1,2,4,6,8,6,4,1]
; AVX-NEXT:    retq
  %cvt = fptoui <8 x float> <float 1.0, float 2.0, float 4.0, float 6.0, float 8.0, float 6.0, float 4.0, float 1.0> to <8 x i32>
  ret <8 x i32> %cvt
}
