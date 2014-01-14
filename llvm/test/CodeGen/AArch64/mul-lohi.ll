; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s

define i128 @test_128bitmul(i128 %lhs, i128 %rhs) {
; CHECK: test_128bitmul:
; CHECK-DAG: umulh [[CARRY:x[0-9]+]], x0, x2
; CHECK-DAG: madd [[PART1:x[0-9]+]], x0, x3, [[CARRY]]
; CHECK: madd x1, x1, x2, [[PART1]]
; CHECK: mul x0, x0, x2

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}
