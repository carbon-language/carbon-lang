; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck %s
@foobar = common global i32 0, align 4

define i32 @foo() {
entry:
  %0 = load i32* @foobar, align 4
  ret i32 %0
}

; CHECK:       [[POFF:\.L[0-9]+\$poff]]:
; CHECK-NEXT:    .long .L.TOC.-[[PB:\.L[0-9]+\$pb]]
; CHECK-NEXT:  foo:
; CHECK:         bl [[PB]]
; CHECK-NEXT:  [[PB]]:
; CHECK:         mflr 30
; CHECK:         lwz [[REG:[0-9]+]], [[POFF]]-[[PB]](30)
; CHECK-NEXT:    add 30, [[REG]], 30
; CHECK:         lwz [[VREG:[0-9]+]], [[VREF:\.LC[0-9]+]]-.L.TOC.(30)
; CHECK:         lwz {{[0-9]+}}, 0([[VREG]])
; CHECK:       [[VREF]]:
; CHECK-NEXT:    .long foobar
