; RUN: opt -S < %s -instcombine | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; Check transforms involving atomic operations

define i32* @test1(i8** %p) {
; CHECK: define i32* @test1
; CHECK: load atomic i8** %p monotonic, align 8
  %c = bitcast i8** %p to i32**
  %r = load atomic i32** %c monotonic, align 8
  ret i32* %r
}

