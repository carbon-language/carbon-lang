; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @add(i32 %arg) {

  ; -- Same size registeres --
  ;CHECK: cost of 1 {{.*}} zext
  %A = zext <4 x i1> undef to <4 x i32>
  ;CHECK: cost of 2 {{.*}} sext
  %B = sext <4 x i1> undef to <4 x i32>
  ;CHECK: cost of 0 {{.*}} trunc
  %C = trunc <4 x i32> undef to <4 x i1>

  ; -- Different size registers --
  ;CHECK-NOT: cost of 1 {{.*}} zext
  %D = zext <8 x i1> undef to <8 x i32>
  ;CHECK-NOT: cost of 2 {{.*}} sext
  %E = sext <8 x i1> undef to <8 x i32>
  ;CHECK-NOT: cost of 2 {{.*}} trunc
  %F = trunc <8 x i32> undef to <8 x i1>

  ; -- scalars --

  ;CHECK: cost of 1 {{.*}} zext
  %G = zext i1 undef to i32
  ;CHECK: cost of 0 {{.*}} trunc
  %H = trunc i32 undef to i1

  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}

define i32 @zext_sext(<8 x i1> %in) {
  ;CHECK: cost of 6 {{.*}} zext
  %Z = zext <8 x i1> %in to <8 x i32>
  ;CHECK: cost of 9 {{.*}} sext
  %S = sext <8 x i1> %in to <8 x i32>

  ;CHECK: cost of 1 {{.*}} sext
  %A = sext <8 x i16> undef to <8 x i32>
  ;CHECK: cost of 1 {{.*}} zext
  %B = zext <8 x i16> undef to <8 x i32>
  ;CHECK: cost of 1 {{.*}} sext
  %C = sext <4 x i32> undef to <4 x i64>
  ;CHECK: cost of 6 {{.*}} sext
  %C1 = sext <4 x i8> undef to <4 x i64>
  ;CHECK: cost of 6 {{.*}} sext
  %C2 = sext <4 x i16> undef to <4 x i64>

  ;CHECK: cost of 1 {{.*}} zext
  %D = zext <4 x i32> undef to <4 x i64>
  ;CHECK: cost of 1 {{.*}} trunc

  %E = trunc <4 x i64> undef to <4 x i32>
  ;CHECK: cost of 1 {{.*}} trunc
  %F = trunc <8 x i32> undef to <8 x i16>

  ;CHECK: cost of 3 {{.*}} trunc
  %G = trunc <8 x i64> undef to <8 x i32>

  ret i32 undef
}

define i32 @masks8(<8 x i1> %in) {
  ;CHECK: cost of 6 {{.*}} zext
  %Z = zext <8 x i1> %in to <8 x i32>
  ;CHECK: cost of 9 {{.*}} sext
  %S = sext <8 x i1> %in to <8 x i32>
  ret i32 undef
}

define i32 @masks4(<4 x i1> %in) {
  ;CHECK: cost of 8 {{.*}} sext
  %S = sext <4 x i1> %in to <4 x i64>
  ret i32 undef
}

define void @sitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
  ; CHECK: cost of 3 {{.*}} sitofp
  %A1 = sitofp <4 x i1> %a to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %A2 = sitofp <4 x i1> %a to <4 x double>

  ; CHECK: cost of 3 {{.*}} sitofp
  %B1 = sitofp <4 x i8> %b to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %B2 = sitofp <4 x i8> %b to <4 x double>

  ; CHECK: cost of 3 {{.*}} sitofp
  %C1 = sitofp <4 x i16> %c to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %C2 = sitofp <4 x i16> %c to <4 x double>

  ; CHECK: cost of 1 {{.*}} sitofp
  %D1 = sitofp <4 x i32> %d to <4 x float>
  ; CHECK: cost of 1 {{.*}} sitofp
  %D2 = sitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @sitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
  ; CHECK: cost of 8 {{.*}} sitofp
  %A1 = sitofp <8 x i1> %a to <8 x float>

  ; CHECK: cost of 8 {{.*}} sitofp
  %B1 = sitofp <8 x i8> %b to <8 x float>

  ; CHECK: cost of 5 {{.*}} sitofp
  %C1 = sitofp <8 x i16> %c to <8 x float>

  ; CHECK: cost of 1 {{.*}} sitofp
  %D1 = sitofp <8 x i32> %d to <8 x float>
  ret void
}

define void @uitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
  ; CHECK: cost of 7 {{.*}} uitofp
  %A1 = uitofp <4 x i1> %a to <4 x float>
  ; CHECK: cost of 7 {{.*}} uitofp
  %A2 = uitofp <4 x i1> %a to <4 x double>

  ; CHECK: cost of 2 {{.*}} uitofp
  %B1 = uitofp <4 x i8> %b to <4 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %B2 = uitofp <4 x i8> %b to <4 x double>

  ; CHECK: cost of 2 {{.*}} uitofp
  %C1 = uitofp <4 x i16> %c to <4 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %C2 = uitofp <4 x i16> %c to <4 x double>

  ; CHECK: cost of 6 {{.*}} uitofp
  %D1 = uitofp <4 x i32> %d to <4 x float>
  ; CHECK: cost of 6 {{.*}} uitofp
  %D2 = uitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @uitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
  ; CHECK: cost of 6 {{.*}} uitofp
  %A1 = uitofp <8 x i1> %a to <8 x float>

  ; CHECK: cost of 5 {{.*}} uitofp
  %B1 = uitofp <8 x i8> %b to <8 x float>

  ; CHECK: cost of 5 {{.*}} uitofp
  %C1 = uitofp <8 x i16> %c to <8 x float>

  ; CHECK: cost of 9 {{.*}} uitofp
  %D1 = uitofp <8 x i32> %d to <8 x float>
  ret void
}
