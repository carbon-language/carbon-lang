; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck -check-prefix=SMALL-BSS %s
@bar = common global i32 0, align 4

define i32 @foo() {
entry:
  %0 = load i32* @bar, align 4
  ret i32 %0
}

!llvm.module.flags = !{!0}
!0 = metadata !{i32 1, metadata !"PIC Level", i32 1}
; SMALL-BSS-LABEL:foo:
; SMALL-BSS:         bl _GLOBAL_OFFSET_TABLE_@local-4
; SMALL-BSS:         mflr 30
; SMALL-BSS:         lwz [[VREG:[0-9]+]], bar@GOT(30)
; SMALL-BSS:         lwz {{[0-9]+}}, 0([[VREG]])
