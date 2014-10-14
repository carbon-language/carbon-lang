; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort -verify-machineinstrs < %s | FileCheck %s

%struct.foo = type { i32, i64, float, double }

define double* @test_struct(%struct.foo* %f) {
; CHECK-LABEL: test_struct
; CHECK:       add x0, x0, #24
  %1 = getelementptr inbounds %struct.foo* %f, i64 0, i32 3
  ret double* %1
}

define i32* @test_array(i32* %a, i64 %i) {
; CHECK-LABEL: test_array
; CHECK:       orr [[REG:x[0-9]+]], xzr, #0x4
; CHECK-NEXT:  madd  x0, x1, [[REG]], x0
  %1 = getelementptr inbounds i32* %a, i64 %i
  ret i32* %1
}
