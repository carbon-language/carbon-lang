; RUN: llc -verify-machineinstrs -O0 < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-AARCH64
; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM64

define i64 @test0() {
; CHECK-LABEL: test0:
; Not produced by move wide instructions, but good to make sure we can return 0 anyway:
; CHECK: mov x0, xzr
  ret i64 0
}

define i64 @test1() {
; CHECK-LABEL: test1:
; CHECK-AARCH64: movz x0, #1
; CHECK-ARM64: orr w0, wzr, #0x1
  ret i64 1
}

define i64 @test2() {
; CHECK-LABEL: test2:
; CHECK-AARCH64: movz x0, #65535
; CHECK-ARM64: orr w0, wzr, #0xffff
  ret i64 65535
}

define i64 @test3() {
; CHECK-LABEL: test3:
; CHECK-AARCH64: movz x0, #1, lsl #16
; CHECK-ARM64: orr w0, wzr, #0x10000
  ret i64 65536
}

define i64 @test4() {
; CHECK-LABEL: test4:
; CHECK-AARCH64: movz x0, #65535, lsl #16
; CHECK-ARM64: orr w0, wzr, #0xffff0000
  ret i64 4294901760
}

define i64 @test5() {
; CHECK-LABEL: test5:
; CHECK-AARCH64: movz x0, #1, lsl #32
; CHECK-ARM64: orr x0, xzr, #0x100000000
  ret i64 4294967296
}

define i64 @test6() {
; CHECK-LABEL: test6:
; CHECK-AARCH64: movz x0, #65535, lsl #32
; CHECK-ARM64: orr x0, xzr, #0xffff00000000
  ret i64 281470681743360
}

define i64 @test7() {
; CHECK-LABEL: test7:
; CHECK-AARCH64: movz x0, #1, lsl #48
; CHECK-ARM64: orr x0, xzr, #0x1000000000000
  ret i64 281474976710656
}

; A 32-bit MOVN can generate some 64-bit patterns that a 64-bit one
; couldn't. Useful even for i64
define i64 @test8() {
; CHECK-LABEL: test8:
; CHECK: movn w0, #60875
  ret i64 4294906420
}

define i64 @test9() {
; CHECK-LABEL: test9:
; CHECK: movn x0, #0
  ret i64 -1
}

define i64 @test10() {
; CHECK-LABEL: test10:
; CHECK: movn x0, #60875, lsl #16
  ret i64 18446744069720047615
}

; For reasonably legitimate reasons returning an i32 results in the
; selection of an i64 constant, so we need a different idiom to test that selection
@var32 = global i32 0

define void @test11() {
; CHECK-LABEL: test11:
; CHECK-AARCH64: mov {{w[0-9]+}}, wzr
; CHECK-ARM64: str wzr
  store i32 0, i32* @var32
  ret void
}

define void @test12() {
; CHECK-LABEL: test12:
; CHECK-AARCH64: movz {{w[0-9]+}}, #1
; CHECK-ARM64: orr {{w[0-9]+}}, wzr, #0x1
  store i32 1, i32* @var32
  ret void
}

define void @test13() {
; CHECK-LABEL: test13:
; CHECK-AARCH64: movz {{w[0-9]+}}, #65535
; CHECK-ARM64: orr {{w[0-9]+}}, wzr, #0xffff
  store i32 65535, i32* @var32
  ret void
}

define void @test14() {
; CHECK-LABEL: test14:
; CHECK-AARCH64: movz {{w[0-9]+}}, #1, lsl #16
; CHECK-ARM64: orr {{w[0-9]+}}, wzr, #0x10000
  store i32 65536, i32* @var32
  ret void
}

define void @test15() {
; CHECK-LABEL: test15:
; CHECK-AARCH64: movz {{w[0-9]+}}, #65535, lsl #16
; CHECK-ARM64: orr {{w[0-9]+}}, wzr, #0xffff0000
  store i32 4294901760, i32* @var32
  ret void
}

define void @test16() {
; CHECK-LABEL: test16:
; CHECK: movn {{w[0-9]+}}, #0
  store i32 -1, i32* @var32
  ret void
}

define i64 @test17() {
; CHECK-LABEL: test17:

  ; Mustn't MOVN w0 here.
; CHECK-AARCH64: movn x0, #2
; CHECK-ARM64: orr x0, xzr, #0xfffffffffffffffd
  ret i64 -3
}
