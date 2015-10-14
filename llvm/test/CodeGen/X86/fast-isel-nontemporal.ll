; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse2 -fast-isel -O0 < %s | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx -fast-isel -O0 < %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX

define void @test_nti32(i32* nocapture %ptr, i32 %X) {
; ALL-LABEL: test_nti32:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    movntil %esi, (%rdi)
; ALL-NEXT:    retq
entry:
  store i32 %X, i32* %ptr, align 4, !nontemporal !1
  ret void
}

define void @test_nti64(i64* nocapture %ptr, i64 %X) {
; ALL-LABEL: test_nti64:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    movntiq %rsi, (%rdi)
; ALL-NEXT:    retq
entry:
  store i64 %X, i64* %ptr, align 8, !nontemporal !1
  ret void
}

define void @test_nt4xfloat(<4 x float>* nocapture %ptr, <4 x float> %X) {
; SSE2-LABEL: test_nt4xfloat:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movntps %xmm0, (%rdi)
; SSE2-NEXT:    retq
;
; AVX-LABEL: test_nt4xfloat:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovntps %xmm0, (%rdi)
; AVX-NEXT:    retq
entry:
  store <4 x float> %X, <4 x float>* %ptr, align 16, !nontemporal !1
  ret void
}

define void @test_nt2xdouble(<2 x double>* nocapture %ptr, <2 x double> %X) {
; SSE2-LABEL: test_nt2xdouble:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movntpd %xmm0, (%rdi)
; SSE2-NEXT:    retq
;
; AVX-LABEL: test_nt2xdouble:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovntpd %xmm0, (%rdi)
; AVX-NEXT:    retq
entry:
  store <2 x double> %X, <2 x double>* %ptr, align 16, !nontemporal !1
  ret void
}

define void @test_nt2xi64(<2 x i64>* nocapture %ptr, <2 x i64> %X) {
; SSE2-LABEL: test_nt2xi64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movntdq %xmm0, (%rdi)
; SSE2-NEXT:    retq
;
; AVX-LABEL: test_nt2xi64:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovntdq %xmm0, (%rdi)
; AVX-NEXT:    retq
entry:
  store <2 x i64> %X, <2 x i64>* %ptr, align 16, !nontemporal !1
  ret void
}

!1 = !{i32 1}
