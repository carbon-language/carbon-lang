; RUN: llc -mtriple=arm64-apple-ios7.0 -mcpu=cyclone %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64_be-linux-gnu -mcpu=cyclone %s -o - | FileCheck --check-prefix=CHECK-BE %s
define i128 @test_128bitmul(i128 %lhs, i128 %rhs) {
; CHECK-LABEL: test_128bitmul:
; CHECK-DAG: mul [[PART1:x[0-9]+]], x0, x3
; CHECK-DAG: umulh [[CARRY:x[0-9]+]], x0, x2
; CHECK: mul [[PART2:x[0-9]+]], x1, x2
; CHECK: mul x0, x0, x2

; CHECK-BE-LABEL: test_128bitmul:
; CHECK-BE-DAG: mul [[PART1:x[0-9]+]], x1, x2
; CHECK-BE-DAG: umulh [[CARRY:x[0-9]+]], x1, x3
; CHECK-BE: mul [[PART2:x[0-9]+]], x0, x3
; CHECK-BE: mul x1, x1, x3

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}
