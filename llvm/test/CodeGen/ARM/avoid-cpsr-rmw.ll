; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a9 | FileCheck %s
; Avoid some 's' 16-bit instruction which partially update CPSR (and add false
; dependency) when it isn't dependent on last CPSR defining instruction.
; rdar://8928208

define i32 @t(i32 %a, i32 %b, i32 %c, i32 %d) nounwind readnone {
 entry:
; CHECK: t:
; CHECK: muls [[REG:(r[0-9]+)]], r2, r3
; CHECK-NEXT: mul  [[REG2:(r[0-9]+)]], r0, r1
; CHECK-NEXT: muls r0, [[REG2]], [[REG]]
  %0 = mul nsw i32 %a, %b
  %1 = mul nsw i32 %c, %d
  %2 = mul nsw i32 %0, %1
  ret i32 %2
}
