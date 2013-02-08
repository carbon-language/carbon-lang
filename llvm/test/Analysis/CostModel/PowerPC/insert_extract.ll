; RUN: opt < %s  -cost-model -analyze -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @insert(i32 %arg) {
  ; CHECK: cost of 13 {{.*}} insertelement
  %x = insertelement <4 x i32> undef, i32 %arg, i32 0
  ret i32 undef
}

define i32 @extract(<4 x i32> %arg) {
  ; CHECK: cost of 13 {{.*}} extractelement
  %x = extractelement <4 x i32> %arg, i32 0
  ret i32 %x
}

