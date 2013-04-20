; RUN: opt < %s -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Make sure that we are not crashing or changing the code.
;CHECK: test
;CHECK: icmp
;CHECK: ret
define void @test(<4 x i32> %in, <4 x i32> %in2) {
  %k = icmp eq <4 x i32> %in, %in2
  ret void
}

