; RUN: llc -mtriple=x86_64-linux -fast-isel -show-mc-encoding < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK:       addl $-2, %eax         # encoding: [0x83,0xc0,0xfe]
define i32 @f(i32* %y) {
  %x = load i32, i32* %y
  %dec = add i32 %x, -2
  ret i32 %dec
}
