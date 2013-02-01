; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@var32 = global i32 0
@var64 = global i64 0

define void @test_zr() {
; CHECK: test_zr:

  store i32 0, i32* @var32
; CHECK: str wzr, [{{x[0-9]+}}, #:lo12:var32]
  store i64 0, i64* @var64
; CHECK: str xzr, [{{x[0-9]+}}, #:lo12:var64]

  ret void
; CHECK: ret
}

define void @test_sp(i32 %val) {
; CHECK: test_sp:

; Important correctness point here is that LLVM doesn't try to use xzr
; as an addressing register: "str w0, [xzr]" is not a valid A64
; instruction (0b11111 in the Rn field would mean "sp").
  %addr = getelementptr i32* null, i64 0
  store i32 %val, i32* %addr
; CHECK: mov x[[NULL:[0-9]+]], xzr
; CHECK: str {{w[0-9]+}}, [x[[NULL]]]

  ret void
; CHECK: ret
}