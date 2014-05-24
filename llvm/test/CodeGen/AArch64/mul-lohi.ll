; RUN: llc -mtriple=arm64-apple-ios7.0 %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64_be-linux-gnu %s -o - | FileCheck --check-prefix=CHECK-BE %s

define i128 @test_128bitmul(i128 %lhs, i128 %rhs) {
; CHECK-LABEL: test_128bitmul:
; CHECK-DAG: umulh [[CARRY:x[0-9]+]], x0, x2
; CHECK-DAG: madd [[PART1:x[0-9]+]], x0, x3, [[CARRY]]
; CHECK: madd x1, x1, x2, [[PART1]]
; CHECK: mul x0, x0, x2

; CHECK-BE-LABEL: test_128bitmul:
; CHECK-BE-DAG: umulh [[CARRY:x[0-9]+]], x1, x3
; CHECK-BE-DAG: madd [[PART1:x[0-9]+]], x1, x2, [[CARRY]]
; CHECK-BE: madd x0, x0, x3, [[PART1]]
; CHECK-BE: mul x1, x1, x3

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}
