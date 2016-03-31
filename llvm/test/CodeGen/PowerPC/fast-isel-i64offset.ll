; RUN: llc -mtriple powerpc64-unknown-linux-gnu -fast-isel -O0 < %s | FileCheck %s

; Verify that pointer offsets larger than 32 bits work correctly.

define void @test(i32* %array) {
; CHECK-LABEL: test:
; CHECK: lis [[REG:[0-9]+]], 16383
; CHECK: ori [[REG]], [[REG]], 65535
; CHECK: sldi [[REG]], [[REG]], 3
; CHECK: stwx {{[0-9]+}}, 3, [[REG]]
  %element = getelementptr i32, i32* %array, i64 2147483646
  store i32 1234, i32* %element
  ret void
}

