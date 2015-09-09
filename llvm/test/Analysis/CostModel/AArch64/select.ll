; RUN: opt < %s  -cost-model -analyze -mtriple=arm64-apple-ios -mcpu=cyclone | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

; CHECK-LABEL: select
define void @select() {
    ; Scalar values
  ; CHECK: cost of 1 {{.*}} select
  %v1 = select i1 undef, i8 undef, i8 undef
  ; CHECK: cost of 1 {{.*}} select
  %v2 = select i1 undef, i16 undef, i16 undef
  ; CHECK: cost of 1 {{.*}} select
  %v3 = select i1 undef, i32 undef, i32 undef
  ; CHECK: cost of 1 {{.*}} select
  %v4 = select i1 undef, i64 undef, i64 undef
  ; CHECK: cost of 1 {{.*}} select
  %v5 = select i1 undef, float undef, float undef
  ; CHECK: cost of 1 {{.*}} select
  %v6 = select i1 undef, double undef, double undef

  ; CHECK: cost of 16 {{.*}} select
  %v13b = select <16 x i1>  undef, <16 x i16> undef, <16 x i16> undef

  ; CHECK: cost of 8 {{.*}} select
  %v15b = select <8 x i1>  undef, <8 x i32> undef, <8 x i32> undef
  ; CHECK: cost of 16 {{.*}} select
  %v15c = select <16 x i1>  undef, <16 x i32> undef, <16 x i32> undef

  ; Vector values - check for vectors of i64s that have a high cost because
  ; they end up scalarized.
  ; CHECK: cost of 80 {{.*}} select
  %v16a = select <4 x i1> undef, <4 x i64> undef, <4 x i64> undef
  ; CHECK: cost of 160 {{.*}} select
  %v16b = select <8 x i1> undef, <8 x i64> undef, <8 x i64> undef
  ; CHECK: cost of 320 {{.*}} select
  %v16c = select <16 x i1> undef, <16 x i64> undef, <16 x i64> undef

  ret void
}
