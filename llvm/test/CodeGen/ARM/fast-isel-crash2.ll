; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=thumbv7-apple-darwin
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=thumbv7-linux-gnueabi
; rdar://9515076
; (Make sure this doesn't crash.)

define i32 @test(i32 %i) {
  %t = trunc i32 %i to i4
  %r = sext i4 %t to i32
  ret i32 %r
}
