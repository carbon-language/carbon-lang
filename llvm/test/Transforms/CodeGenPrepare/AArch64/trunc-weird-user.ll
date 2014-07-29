; RUN: opt -S -codegenprepare -mtriple=arm64-apple-ios7.0 %s | FileCheck %s

%foo = type { i8 }

define %foo @test_merge(i32 %in) {
; CHECK-LABEL: @test_merge

  ; CodeGenPrepare was requesting the EVT for { i8 } to determine
  ; whether the insertvalue user of the trunc was legal. This
  ; asserted.

; CHECK: insertvalue %foo undef, i8 %byte, 0
  %lobit = lshr i32 %in, 31
  %byte = trunc i32 %lobit to i8
  %struct = insertvalue %foo undef, i8 %byte, 0
  ret %"foo" %struct
}
