; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-objdump -s - \
; RUN:   | FileCheck %s

define void @test() nounwind {
entry:
  ret void
}

; CHECK: section .text
; CHECK: section .ARM.exidx
; CHECK-NEXT: 0000 00000000 01000000
