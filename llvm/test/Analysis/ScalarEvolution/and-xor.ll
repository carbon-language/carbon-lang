; RUN: opt < %s -scalar-evolution -analyze | FileCheck %s

; CHECK: -->  (zext
; CHECK: -->  (zext
; CHECK-NOT: -->  (zext

define i32 @foo(i32 %x) {
  %n = and i32 %x, 255
  %y = xor i32 %n, 255
  ret i32 %y
}
