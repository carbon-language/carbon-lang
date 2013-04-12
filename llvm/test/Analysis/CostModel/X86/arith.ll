; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 | FileCheck %s --check-prefix=SSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core-avx2 | FileCheck %s --check-prefix=AVX2

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @add(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} add
  %A = add <4 x i32> undef, undef
  ;CHECK: cost of 4 {{.*}} add
  %B = add <8 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} add
  %C = add <2 x i64> undef, undef
  ;CHECK: cost of 4 {{.*}} add
  %D = add <4 x i64> undef, undef
  ;CHECK: cost of 8 {{.*}} add
  %E = add <8 x i64> undef, undef
  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}


define i32 @xor(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} xor
  %A = xor <4 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %B = xor <8 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %C = xor <2 x i64> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %D = xor <4 x i64> undef, undef
  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}

; CHECK: mul
define void @mul() {
  ; A <2 x i32> gets expanded to a <2 x i64> vector.
  ; A <2 x i64> vector multiply is implemented using
  ; 3 PMULUDQ and 2 PADDS and 4 shifts.
  ;CHECK: cost of 9 {{.*}} mul
  %A0 = mul <2 x i32> undef, undef
  ;CHECK: cost of 9 {{.*}} mul
  %A1 = mul <2 x i64> undef, undef
  ;CHECK: cost of 18 {{.*}} mul
  %A2 = mul <4 x i64> undef, undef
  ret void
}

; SSE3: sse3mull
define void @sse3mull() {
  ; SSE3: cost of 6 {{.*}} mul
  %A0 = mul <4 x i32> undef, undef
  ret void
  ; SSE3: avx2mull
}

; AVX2: avx2mull
define void @avx2mull() {
  ; AVX2: cost of 9 {{.*}} mul
  %A0 = mul <4 x i64> undef, undef
  ret void
  ; AVX2: fmul
}

; CHECK: fmul
define i32 @fmul(i32 %arg) {
  ;CHECK: cost of 2 {{.*}} fmul
  %A = fmul <4 x float> undef, undef
  ;CHECK: cost of 2 {{.*}} fmul
  %B = fmul <8 x float> undef, undef
  ret i32 undef
}

; AVX: shift
; AVX2: shift
define void @shift() {
  ; AVX: cost of 2 {{.*}} shl
  ; AVX2: cost of 1 {{.*}} shl
  %A0 = shl <4 x i32> undef, undef
  ; AVX: cost of 2 {{.*}} shl
  ; AVX2: cost of 1 {{.*}} shl
  %A1 = shl <2 x i64> undef, undef

  ; AVX: cost of 2 {{.*}} lshr
  ; AVX2: cost of 1 {{.*}} lshr
  %B0 = lshr <4 x i32> undef, undef
  ; AVX: cost of 2 {{.*}} lshr
  ; AVX2: cost of 1 {{.*}} lshr
  %B1 = lshr <2 x i64> undef, undef

  ; AVX: cost of 2 {{.*}} ashr
  ; AVX2: cost of 1 {{.*}} ashr
  %C0 = ashr <4 x i32> undef, undef
  ; AVX: cost of 6 {{.*}} ashr
  ; AVX2: cost of 20 {{.*}} ashr
  %C1 = ashr <2 x i64> undef, undef

  ret void
}

; AVX: avx2shift
; AVX2: avx2shift
define void @avx2shift() {
  ; AVX: cost of 2 {{.*}} shl
  ; AVX2: cost of 1 {{.*}} shl
  %A0 = shl <8 x i32> undef, undef
  ; AVX: cost of 2 {{.*}} shl
  ; AVX2: cost of 1 {{.*}} shl
  %A1 = shl <4 x i64> undef, undef

  ; AVX: cost of 2 {{.*}} lshr
  ; AVX2: cost of 1 {{.*}} lshr
  %B0 = lshr <8 x i32> undef, undef
  ; AVX: cost of 2 {{.*}} lshr
  ; AVX2: cost of 1 {{.*}} lshr
  %B1 = lshr <4 x i64> undef, undef

  ; AVX: cost of 2 {{.*}} ashr
  ; AVX2: cost of 1 {{.*}} ashr
  %C0 = ashr <8 x i32> undef, undef
  ; AVX: cost of 12 {{.*}} ashr
  ; AVX2: cost of 40 {{.*}} ashr
  %C1 = ashr <4 x i64> undef, undef

  ret void
}
