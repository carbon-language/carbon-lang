; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'fadd'
define i32 @fadd(i32 %arg) {
  ; SSSE3: cost of 2 {{.*}} %A = fadd
  ; SSE42: cost of 2 {{.*}} %A = fadd
  ; AVX: cost of 2 {{.*}} %A = fadd
  ; AVX2: cost of 2 {{.*}} %A = fadd
  ; AVX512: cost of 2 {{.*}} %A = fadd
  %A = fadd <4 x float> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B = fadd
  ; SSE42: cost of 4 {{.*}} %B = fadd
  ; AVX: cost of 2 {{.*}} %B = fadd
  ; AVX2: cost of 2 {{.*}} %B = fadd
  ; AVX512: cost of 2 {{.*}} %B = fadd
  %B = fadd <8 x float> undef, undef
  ; SSSE3: cost of 8 {{.*}} %C = fadd
  ; SSE42: cost of 8 {{.*}} %C = fadd
  ; AVX: cost of 4 {{.*}} %C = fadd
  ; AVX2: cost of 4 {{.*}} %C = fadd
  ; AVX512: cost of 2 {{.*}} %C = fadd
  %C = fadd <16 x float> undef, undef

  ; SSSE3: cost of 2 {{.*}} %D = fadd
  ; SSE42: cost of 2 {{.*}} %D = fadd
  ; AVX: cost of 2 {{.*}} %D = fadd
  ; AVX2: cost of 2 {{.*}} %D = fadd
  ; AVX512: cost of 2 {{.*}} %D = fadd
  %D = fadd <2 x double> undef, undef
  ; SSSE3: cost of 4 {{.*}} %E = fadd
  ; SSE42: cost of 4 {{.*}} %E = fadd
  ; AVX: cost of 2 {{.*}} %E = fadd
  ; AVX2: cost of 2 {{.*}} %E = fadd
  ; AVX512: cost of 2 {{.*}} %E = fadd
  %E = fadd <4 x double> undef, undef
  ; SSSE3: cost of 8 {{.*}} %F = fadd
  ; SSE42: cost of 8 {{.*}} %F = fadd
  ; AVX: cost of 4 {{.*}} %F = fadd
  ; AVX2: cost of 4 {{.*}} %F = fadd
  ; AVX512: cost of 2 {{.*}} %F = fadd
  %F = fadd <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fsub'
define i32 @fsub(i32 %arg) {
  ; SSSE3: cost of 2 {{.*}} %A = fsub
  ; SSE42: cost of 2 {{.*}} %A = fsub
  ; AVX: cost of 2 {{.*}} %A = fsub
  ; AVX2: cost of 2 {{.*}} %A = fsub
  ; AVX512: cost of 2 {{.*}} %A = fsub
  %A = fsub <4 x float> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B = fsub
  ; SSE42: cost of 4 {{.*}} %B = fsub
  ; AVX: cost of 2 {{.*}} %B = fsub
  ; AVX2: cost of 2 {{.*}} %B = fsub
  ; AVX512: cost of 2 {{.*}} %B = fsub
  %B = fsub <8 x float> undef, undef
  ; SSSE3: cost of 8 {{.*}} %C = fsub
  ; SSE42: cost of 8 {{.*}} %C = fsub
  ; AVX: cost of 4 {{.*}} %C = fsub
  ; AVX2: cost of 4 {{.*}} %C = fsub
  ; AVX512: cost of 2 {{.*}} %C = fsub
  %C = fsub <16 x float> undef, undef

  ; SSSE3: cost of 2 {{.*}} %D = fsub
  ; SSE42: cost of 2 {{.*}} %D = fsub
  ; AVX: cost of 2 {{.*}} %D = fsub
  ; AVX2: cost of 2 {{.*}} %D = fsub
  ; AVX512: cost of 2 {{.*}} %D = fsub
  %D = fsub <2 x double> undef, undef
  ; SSSE3: cost of 4 {{.*}} %E = fsub
  ; SSE42: cost of 4 {{.*}} %E = fsub
  ; AVX: cost of 2 {{.*}} %E = fsub
  ; AVX2: cost of 2 {{.*}} %E = fsub
  ; AVX512: cost of 2 {{.*}} %E = fsub
  %E = fsub <4 x double> undef, undef
  ; SSSE3: cost of 8 {{.*}} %F = fsub
  ; SSE42: cost of 8 {{.*}} %F = fsub
  ; AVX: cost of 4 {{.*}} %F = fsub
  ; AVX2: cost of 4 {{.*}} %F = fsub
  ; AVX512: cost of 2 {{.*}} %F = fsub
  %F = fsub <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fmul'
define i32 @fmul(i32 %arg) {
  ; SSSE3: cost of 2 {{.*}} %A = fmul
  ; SSE42: cost of 2 {{.*}} %A = fmul
  ; AVX: cost of 2 {{.*}} %A = fmul
  ; AVX2: cost of 2 {{.*}} %A = fmul
  ; AVX512: cost of 2 {{.*}} %A = fmul
  %A = fmul <4 x float> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B = fmul
  ; SSE42: cost of 4 {{.*}} %B = fmul
  ; AVX: cost of 2 {{.*}} %B = fmul
  ; AVX2: cost of 2 {{.*}} %B = fmul
  ; AVX512: cost of 2 {{.*}} %B = fmul
  %B = fmul <8 x float> undef, undef
  ; SSSE3: cost of 8 {{.*}} %C = fmul
  ; SSE42: cost of 8 {{.*}} %C = fmul
  ; AVX: cost of 4 {{.*}} %C = fmul
  ; AVX2: cost of 4 {{.*}} %C = fmul
  ; AVX512: cost of 2 {{.*}} %C = fmul
  %C = fmul <16 x float> undef, undef

  ; SSSE3: cost of 2 {{.*}} %D = fmul
  ; SSE42: cost of 2 {{.*}} %D = fmul
  ; AVX: cost of 2 {{.*}} %D = fmul
  ; AVX2: cost of 2 {{.*}} %D = fmul
  ; AVX512: cost of 2 {{.*}} %D = fmul
  %D = fmul <2 x double> undef, undef
  ; SSSE3: cost of 4 {{.*}} %E = fmul
  ; SSE42: cost of 4 {{.*}} %E = fmul
  ; AVX: cost of 2 {{.*}} %E = fmul
  ; AVX2: cost of 2 {{.*}} %E = fmul
  ; AVX512: cost of 2 {{.*}} %E = fmul
  %E = fmul <4 x double> undef, undef
  ; SSSE3: cost of 8 {{.*}} %F = fmul
  ; SSE42: cost of 8 {{.*}} %F = fmul
  ; AVX: cost of 4 {{.*}} %F = fmul
  ; AVX2: cost of 4 {{.*}} %F = fmul
  ; AVX512: cost of 2 {{.*}} %F = fmul
  %F = fmul <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fdiv'
define i32 @fdiv(i32 %arg) {
  ; SSSE3: cost of 2 {{.*}} %A = fdiv
  ; SSE42: cost of 2 {{.*}} %A = fdiv
  ; AVX: cost of 2 {{.*}} %A = fdiv
  ; AVX2: cost of 2 {{.*}} %A = fdiv
  ; AVX512: cost of 2 {{.*}} %A = fdiv
  %A = fdiv <4 x float> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B = fdiv
  ; SSE42: cost of 4 {{.*}} %B = fdiv
  ; AVX: cost of 2 {{.*}} %B = fdiv
  ; AVX2: cost of 2 {{.*}} %B = fdiv
  ; AVX512: cost of 2 {{.*}} %B = fdiv
  %B = fdiv <8 x float> undef, undef
  ; SSSE3: cost of 8 {{.*}} %C = fdiv
  ; SSE42: cost of 8 {{.*}} %C = fdiv
  ; AVX: cost of 4 {{.*}} %C = fdiv
  ; AVX2: cost of 4 {{.*}} %C = fdiv
  ; AVX512: cost of 2 {{.*}} %C = fdiv
  %C = fdiv <16 x float> undef, undef

  ; SSSE3: cost of 2 {{.*}} %D = fdiv
  ; SSE42: cost of 2 {{.*}} %D = fdiv
  ; AVX: cost of 2 {{.*}} %D = fdiv
  ; AVX2: cost of 2 {{.*}} %D = fdiv
  ; AVX512: cost of 2 {{.*}} %D = fdiv
  %D = fdiv <2 x double> undef, undef
  ; SSSE3: cost of 4 {{.*}} %E = fdiv
  ; SSE42: cost of 4 {{.*}} %E = fdiv
  ; AVX: cost of 2 {{.*}} %E = fdiv
  ; AVX2: cost of 2 {{.*}} %E = fdiv
  ; AVX512: cost of 2 {{.*}} %E = fdiv
  %E = fdiv <4 x double> undef, undef
  ; SSSE3: cost of 8 {{.*}} %F = fdiv
  ; SSE42: cost of 8 {{.*}} %F = fdiv
  ; AVX: cost of 4 {{.*}} %F = fdiv
  ; AVX2: cost of 4 {{.*}} %F = fdiv
  ; AVX512: cost of 2 {{.*}} %F = fdiv
  %F = fdiv <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'frem'
define i32 @frem(i32 %arg) {
  ; SSSE3: cost of 14 {{.*}} %A = frem
  ; SSE42: cost of 14 {{.*}} %A = frem
  ; AVX: cost of 14 {{.*}} %A = frem
  ; AVX2: cost of 14 {{.*}} %A = frem
  ; AVX512: cost of 14 {{.*}} %A = frem
  %A = frem <4 x float> undef, undef
  ; SSSE3: cost of 28 {{.*}} %B = frem
  ; SSE42: cost of 28 {{.*}} %B = frem
  ; AVX: cost of 30 {{.*}} %B = frem
  ; AVX2: cost of 30 {{.*}} %B = frem
  ; AVX512: cost of 30 {{.*}} %B = frem
  %B = frem <8 x float> undef, undef
  ; SSSE3: cost of 56 {{.*}} %C = frem
  ; SSE42: cost of 56 {{.*}} %C = frem
  ; AVX: cost of 60 {{.*}} %C = frem
  ; AVX2: cost of 60 {{.*}} %C = frem
  ; AVX512: cost of 62 {{.*}} %C = frem
  %C = frem <16 x float> undef, undef

  ; SSSE3: cost of 6 {{.*}} %D = frem
  ; SSE42: cost of 6 {{.*}} %D = frem
  ; AVX: cost of 6 {{.*}} %D = frem
  ; AVX2: cost of 6 {{.*}} %D = frem
  ; AVX512: cost of 6 {{.*}} %D = frem
  %D = frem <2 x double> undef, undef
  ; SSSE3: cost of 12 {{.*}} %E = frem
  ; SSE42: cost of 12 {{.*}} %E = frem
  ; AVX: cost of 14 {{.*}} %E = frem
  ; AVX2: cost of 14 {{.*}} %E = frem
  ; AVX512: cost of 14 {{.*}} %E = frem
  %E = frem <4 x double> undef, undef
  ; SSSE3: cost of 24 {{.*}} %F = frem
  ; SSE42: cost of 24 {{.*}} %F = frem
  ; AVX: cost of 28 {{.*}} %F = frem
  ; AVX2: cost of 28 {{.*}} %F = frem
  ; AVX512: cost of 30 {{.*}} %F = frem
  %F = frem <8 x double> undef, undef

  ret i32 undef
}
