; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -o - %s | FileCheck %s

define i32 @divide(i32 %i, i32 %j) nounwind {
entry:
  %quotient = sdiv i32 %i, %j
  ret i32 %quotient
}

; CHECK-NOT: __aeabi_idiv

