; RUN: llc -O3 -mtriple=x86_64-unknown -mcpu=x86-64 -mattr=+sse2 < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -O3 -mtriple=x86_64-unknown -mcpu=x86-64 -mattr=+avx2 < %s | FileCheck %s --check-prefix=AVX

;
; Float Comparisons
; Only equal/not-equal/ordered/unordered can be safely commuted
;

define <4 x i32> @commute_cmpps_eq(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_eq
  ;SSE:       cmpeqps (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_eq
  ;AVX:       vcmpeqps (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp oeq <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <4 x i32> @commute_cmpps_ne(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_ne
  ;SSE:       cmpneqps (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_ne
  ;AVX:       vcmpneqps (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp une <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <4 x i32> @commute_cmpps_ord(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_ord
  ;SSE:       cmpordps (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_ord
  ;AVX:       vcmpordps (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp ord <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <4 x i32> @commute_cmpps_uno(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_uno
  ;SSE:       cmpunordps (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_uno
  ;AVX:       vcmpunordps (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp uno <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <4 x i32> @commute_cmpps_lt(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_lt
  ;SSE:       movaps (%rdi), %xmm1
  ;SSE-NEXT:  cmpltps %xmm0, %xmm1
  ;SSE-NEXT:  movaps %xmm1, %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_lt
  ;AVX:       vmovaps (%rdi), %xmm1
  ;AVX-NEXT:  vcmpltps %xmm0, %xmm1, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp olt <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <4 x i32> @commute_cmpps_le(<4 x float>* %a0, <4 x float> %a1) #0 {
  ;SSE-LABEL: commute_cmpps_le
  ;SSE:       movaps (%rdi), %xmm1
  ;SSE-NEXT:  cmpleps %xmm0, %xmm1
  ;SSE-NEXT:  movaps %xmm1, %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmpps_le
  ;AVX:       vmovaps (%rdi), %xmm1
  ;AVX-NEXT:  vcmpleps %xmm0, %xmm1, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <4 x float>, <4 x float>* %a0
  %2 = fcmp ole <4 x float> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i32> @commute_cmpps_eq_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_eq_ymm
  ;AVX:       vcmpeqps (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp oeq <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <8 x i32> @commute_cmpps_ne_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_ne_ymm
  ;AVX:       vcmpneqps (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp une <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <8 x i32> @commute_cmpps_ord_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_ord_ymm
  ;AVX:       vcmpordps (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp ord <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <8 x i32> @commute_cmpps_uno_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_uno_ymm
  ;AVX:       vcmpunordps (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp uno <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <8 x i32> @commute_cmpps_lt_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_lt_ymm
  ;AVX:       vmovaps (%rdi), %ymm1
  ;AVX-NEXT:  vcmpltps %ymm0, %ymm1, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp olt <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <8 x i32> @commute_cmpps_le_ymm(<8 x float>* %a0, <8 x float> %a1) #0 {
  ;AVX-LABEL: commute_cmpps_le_ymm
  ;AVX:       vmovaps (%rdi), %ymm1
  ;AVX-NEXT:  vcmpleps %ymm0, %ymm1, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <8 x float>, <8 x float>* %a0
  %2 = fcmp ole <8 x float> %1, %a1
  %3 = sext <8 x i1> %2 to <8 x i32>
  ret <8 x i32> %3
}

;
; Double Comparisons
; Only equal/not-equal/ordered/unordered can be safely commuted
;

define <2 x i64> @commute_cmppd_eq(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_eq
  ;SSE:       cmpeqpd (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_eq
  ;AVX:       vcmpeqpd (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp oeq <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @commute_cmppd_ne(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_ne
  ;SSE:       cmpneqpd (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_ne
  ;AVX:       vcmpneqpd (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp une <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @commute_cmppd_ord(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_ord
  ;SSE:       cmpordpd (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_ord
  ;AVX:       vcmpordpd (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp ord <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @commute_cmppd_uno(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_uno
  ;SSE:       cmpunordpd (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_uno
  ;AVX:       vcmpunordpd (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp uno <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @commute_cmppd_lt(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_lt
  ;SSE:       movapd (%rdi), %xmm1
  ;SSE-NEXT:  cmpltpd %xmm0, %xmm1
  ;SSE-NEXT:  movapd %xmm1, %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_lt
  ;AVX:       vmovapd (%rdi), %xmm1
  ;AVX-NEXT:  vcmpltpd %xmm0, %xmm1, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp olt <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @commute_cmppd_le(<2 x double>* %a0, <2 x double> %a1) #0 {
  ;SSE-LABEL: commute_cmppd_le
  ;SSE:       movapd (%rdi), %xmm1
  ;SSE-NEXT:  cmplepd %xmm0, %xmm1
  ;SSE-NEXT:  movapd %xmm1, %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_cmppd_le
  ;AVX:       vmovapd (%rdi), %xmm1
  ;AVX-NEXT:  vcmplepd %xmm0, %xmm1, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x double>, <2 x double>* %a0
  %2 = fcmp ole <2 x double> %1, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @commute_cmppd_eq_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_eq
  ;AVX:       vcmpeqpd (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp oeq <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @commute_cmppd_ne_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_ne
  ;AVX:       vcmpneqpd (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp une <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @commute_cmppd_ord_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_ord
  ;AVX:       vcmpordpd (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp ord <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @commute_cmppd_uno_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_uno
  ;AVX:       vcmpunordpd (%rdi), %ymm0, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp uno <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @commute_cmppd_lt_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_lt
  ;AVX:       vmovapd (%rdi), %ymm1
  ;AVX-NEXT:  vcmpltpd %ymm0, %ymm1, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp olt <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @commute_cmppd_le_ymmm(<4 x double>* %a0, <4 x double> %a1) #0 {
  ;AVX-LABEL: commute_cmppd_le
  ;AVX:       vmovapd (%rdi), %ymm1
  ;AVX-NEXT:  vcmplepd %ymm0, %ymm1, %ymm0
  ;AVX-NEXT:  retq

  %1 = load <4 x double>, <4 x double>* %a0
  %2 = fcmp ole <4 x double> %1, %a1
  %3 = sext <4 x i1> %2 to <4 x i64>
  ret <4 x i64> %3
}
