; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s

@var = global i1 0

define i32 @test_sextloadi32() {
; CHECK-LABEL: test_sextloadi32

  %val = load i1* @var
  %ret = sext i1 %val to i32
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]
; CHECK: {{sbfx x[0-9]+, x[0-9]+, #0, #1|sbfm w[0-9]+, w[0-9]+, #0, #0}}

  ret i32 %ret
; CHECK: ret
}

define i64 @test_sextloadi64() {
; CHECK-LABEL: test_sextloadi64

  %val = load i1* @var
  %ret = sext i1 %val to i64
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]
; CHECK: {{sbfx x[0-9]+, x[0-9]+, #0, #1|sbfm x[0-9]+, x[0-9]+, #0, #0}}

  ret i64 %ret
; CHECK: ret
}

define i32 @test_zextloadi32() {
; CHECK-LABEL: test_zextloadi32

; It's not actually necessary that "ret" is next, but as far as LLVM
; is concerned only 0 or 1 should be loadable so no extension is
; necessary.
  %val = load i1* @var
  %ret = zext i1 %val to i32
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]

  ret i32 %ret
; CHECK-NEXT: ret
}

define i64 @test_zextloadi64() {
; CHECK-LABEL: test_zextloadi64

; It's not actually necessary that "ret" is next, but as far as LLVM
; is concerned only 0 or 1 should be loadable so no extension is
; necessary.
  %val = load i1* @var
  %ret = zext i1 %val to i64
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]

  ret i64 %ret
; CHECK-NEXT: ret
}
