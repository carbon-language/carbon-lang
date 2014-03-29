; RUN: llc -O0 -mtriple=arm64-none-linux-gnu -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s

; If the .tlsdesccall and blr parts are emitted completely separately (even with
; glue) then LLVM will separate them quite happily (with a spill at O0, hence
; the option). This is definitely wrong, so we make sure they are emitted
; together.

@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
; CHECK-LABEL: test_generaldynamic:

  %val = load i32* @general_dynamic_var
  ret i32 %val

; CHECK: .tlsdesccall general_dynamic_var
; CHECK-NEXT: blr {{x[0-9]+}}
}
