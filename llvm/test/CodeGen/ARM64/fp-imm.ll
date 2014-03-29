; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

; CHECK: literal8
; CHECK: .quad  4614256656552045848
define double @foo() {
; CHECK: _foo:
; CHECK: adrp x[[REG:[0-9]+]], lCPI0_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI0_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x400921FB54442D18
}

; CHECK: literal4
; CHECK: .long 1078530011
define float @bar() {
; CHECK: _bar:
; CHECK:  adrp  x[[REG:[0-9]+]], lCPI1_0@PAGE
; CHECK:  ldr s0, [x[[REG]], lCPI1_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret float 0x400921FB60000000
}
