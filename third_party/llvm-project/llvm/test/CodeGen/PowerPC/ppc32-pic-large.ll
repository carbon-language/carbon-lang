; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck --check-prefixes=LARGE,LARGE-BSS %s
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mattr=+secure-plt -relocation-model=pic | FileCheck --check-prefixes=LARGE,LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-unknown-netbsd -mattr=+secure-plt -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-unknown-netbsd -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-unknown-openbsd -mattr=+secure-plt -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-unknown-openbsd -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-linux-musl -mattr=+secure-plt -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
; RUN: llc < %s -mtriple=powerpc-linux-musl -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
$bar1 = comdat any

@bar = common global i32 0, align 4
@bar1 = global i32 0, align 4, comdat($bar1)
@bar2 = global i32 0, align 4, comdat($bar1)

declare i32 @call_foo(i32, ...)

define i32 @foo() {
entry:
  %0 = load i32, i32* @bar, align 4
  %call = call i32 (i32, ...) @call_foo(i32 %0, i32 0, i32 1, i32 2, i32 4, i32 8, i32 16, i32 32, i32 64)
  ret i32 %0
}

define i32 @load() {
entry:
  %0 = load i32, i32* @bar1
  %1 = load i32, i32* @bar2
  %2 = add i32 %0, %1
  ret i32 %2
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
; LARGE-BSS:       [[POFF:\.L[0-9]+\$poff]]:
; LARGE-BSS-NEXT:    .long .LTOC-[[PB:\.L[0-9]+\$pb]]
; LARGE-BSS-NEXT:  foo:
; LARGE-BSS:         stwu 1, -32(1)
; LARGE-BSS:         stw 30, 24(1)
; LARGE-BSS:         bl [[PB]]
; LARGE-BSS-NEXT:  [[PB]]:
; LARGE-BSS:         mflr 30
; LARGE-BSS:         lwz [[REG:[0-9]+]], [[POFF]]-[[PB]](30)
; LARGE-BSS-NEXT:    add 30, [[REG]], 30
; LARGE-BSS-DAG:     lwz [[VREG:[0-9]+]], [[VREF:\.LC[0-9]+]]-.LTOC(30)
; LARGE-BSS-DAG:     lwz {{[0-9]+}}, 0([[VREG]])
; LARGE-BSS-DAG:     stw {{[0-9]+}}, 8(1)
; LARGE-BSS:         lwz 30, 24(1)
; LARGE-SECUREPLT:   addis 30, 30, .LTOC-.L0$pb@ha
; LARGE-SECUREPLT:   addi 30, 30, .LTOC-.L0$pb@l
; LARGE-SECUREPLT:   bl call_foo@PLT+32768

; LARGE:      .section .bss.bar1,"aGw",@nobits,bar1,comdat
; LARGE:      bar1:
; LARGE:      .section .bss.bar2,"aGw",@nobits,bar1,comdat
; LARGE:      bar2:
; LARGE:      .section .got2,"aw",@progbits
; LARGE-NEXT: .p2align 2
; LARGE-NEXT: .LC0:
; LARGE-NEXT:  .long bar
; LARGE-NEXT: .LC1:
; LARGE-NEXT:  .long bar1
; LARGE-NEXT: .LC2:
; LARGE-NEXT:  .long bar2
