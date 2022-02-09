; RUN: llc -mtriple arm64-ios- %s -o - | FileCheck %s

; Check the -8 constant is shrunk if there are multiple users of the AND instruction.

; CHECK-LABEL:  _test:
; CHECK:          and x0, x0, #0xfffffff8
; CHECK-NEXT:     add x19, x0, #10
; CHECK-NEXT:     bl  _user

define i64 @test(i32 %a) {
  %ext = zext i32 %a to i64
  %v1 = and i64 %ext, -8
  %v2 = add i64 %v1, 10
  call void @user(i64 %v1)
  ret i64 %v2
}

declare void @user(i64)
