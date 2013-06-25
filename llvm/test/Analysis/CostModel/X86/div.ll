; RUN: opt -mtriple=x86_64-apple-darwin -mcpu=core2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX2 %s


define void @div_sse() {
  ; SSE2: div_sse
  ; SSE2: cost of 320 {{.*}} sdiv
  %a0 = sdiv <16 x i8> undef, undef
  ; SSE2: cost of 160 {{.*}} sdiv
  %a1 = sdiv <8 x i16> undef, undef
  ; SSE2: cost of 80 {{.*}} sdiv
  %a2 = sdiv <4 x i32> undef, undef
  ; SSE2: cost of 40 {{.*}} sdiv
  %a3 = sdiv <2 x i32> undef, undef
  ret void
}
; SSE2: div_avx

define void @div_avx() {
  ; AVX2: div_avx
  ; AVX2: cost of 640 {{.*}} sdiv
  %a0 = sdiv <32 x i8> undef, undef
  ; AVX2: cost of 320 {{.*}} sdiv
  %a1 = sdiv <16 x i16> undef, undef
  ; AVX2: cost of 160 {{.*}} sdiv
  %a2 = sdiv <8 x i32> undef, undef
  ; AVX2: cost of 80 {{.*}} sdiv
  %a3 = sdiv <4 x i32> undef, undef
  ret void
}


