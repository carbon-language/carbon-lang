; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck -check-prefix=LARGE-BSS %s
@bar = common global i32 0, align 4

define i32 @foo() {
entry:
  %0 = load i32* @bar, align 4
  ret i32 %0
}

!llvm.module.flags = !{!0}
!0 = metadata !{i32 1, metadata !"PIC Level", i32 2}
; LARGE-BSS:       [[POFF:\.L[0-9]+\$poff]]:
; LARGE-BSS-NEXT:    .long .LTOC-[[PB:\.L[0-9]+\$pb]]
; LARGE-BSS-NEXT:  foo:
; LARGE-BSS:         bl [[PB]]
; LARGE-BSS-NEXT:  [[PB]]:
; LARGE-BSS:         mflr 30
; LARGE-BSS:         lwz [[REG:[0-9]+]], [[POFF]]-[[PB]](30)
; LARGE-BSS-NEXT:    add 30, [[REG]], 30
; LARGE-BSS:         lwz [[VREG:[0-9]+]], [[VREF:\.LC[0-9]+]]-.LTOC(30)
; LARGE-BSS:         lwz {{[0-9]+}}, 0([[VREG]])
; LARGE-BSS:       [[VREF]]:
; LARGE-BSS-NEXT:    .long bar
