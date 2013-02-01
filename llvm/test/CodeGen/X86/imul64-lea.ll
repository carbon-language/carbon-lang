; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 | FileCheck %s

; Test that 64-bit LEAs are generated for both LP64 and ILP32 in 64-bit mode.
declare i64 @foo64()

define i64 @test64() {
  %tmp.0 = tail call i64 @foo64( )
  %tmp.1 = mul i64 %tmp.0, 9
; CHECK-NOT: mul
; CHECK: leaq
  ret i64 %tmp.1
}

; Test that 32-bit LEAs are generated for both LP64 and ILP32 in 64-bit mode.
declare i32 @foo32()

define i32 @test32() {
  %tmp.0 = tail call i32 @foo32( )
  %tmp.1 = mul i32 %tmp.0, 9
; CHECK-NOT: mul
; CHECK: leal
  ret i32 %tmp.1
}

