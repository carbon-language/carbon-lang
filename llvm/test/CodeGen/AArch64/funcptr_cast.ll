; RUN: llc < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

define i8 @test() {
; CHECK-LABEL: @test
; CHECK: adrp {{x[0-9]+}}, foo
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, :lo12:foo
; CHECK: ldrb w0, [{{x[0-9]+}}]
entry:
  %0 = load i8, i8* bitcast (void (...)* @foo to i8*), align 1
  ret i8 %0
}

declare void @foo(...)
