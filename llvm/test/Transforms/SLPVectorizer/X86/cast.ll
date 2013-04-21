; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; int foo(int * restrict A, char * restrict B) {
;     A[0] = B[0];
;     A[1] = B[1];
;     A[2] = B[2];
;     A[3] = B[3];
; }
;CHECK: @foo
;CHECK: load <4 x i8>
;CHECK: sext
;CHECK: store <4 x i32>
define i32 @foo(i32* noalias nocapture %A, i8* noalias nocapture %B) {
entry:
  %0 = load i8* %B, align 1
  %conv = sext i8 %0 to i32
  store i32 %conv, i32* %A, align 4
  %arrayidx2 = getelementptr inbounds i8* %B, i64 1
  %1 = load i8* %arrayidx2, align 1
  %conv3 = sext i8 %1 to i32
  %arrayidx4 = getelementptr inbounds i32* %A, i64 1
  store i32 %conv3, i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i8* %B, i64 2
  %2 = load i8* %arrayidx5, align 1
  %conv6 = sext i8 %2 to i32
  %arrayidx7 = getelementptr inbounds i32* %A, i64 2
  store i32 %conv6, i32* %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i8* %B, i64 3
  %3 = load i8* %arrayidx8, align 1
  %conv9 = sext i8 %3 to i32
  %arrayidx10 = getelementptr inbounds i32* %A, i64 3
  store i32 %conv9, i32* %arrayidx10, align 4
  ret i32 undef
}

