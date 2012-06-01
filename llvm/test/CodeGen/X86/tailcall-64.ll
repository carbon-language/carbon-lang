; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.4.0"

declare i64 @testi()

define i64 @test_trivial() {
 %A = tail call i64 @testi()
 ret i64 %A
}
; CHECK: test_trivial:
; CHECK: jmp	_testi                  ## TAILCALL


define i64 @test_noop_bitcast() {
 %A = tail call i64 @testi()
 %B = bitcast i64 %A to i64
 ret i64 %B
}
; CHECK: test_noop_bitcast:
; CHECK: jmp	_testi                  ## TAILCALL
