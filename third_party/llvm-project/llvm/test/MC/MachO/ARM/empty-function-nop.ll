; RUN: llc < %s -filetype=obj -mtriple=thumbv6-apple-darwin -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-T1 %s
; RUN: llc < %s -filetype=obj -mtriple=thumbv7-apple-darwin -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-T2 %s
; RUN: llc < %s -filetype=obj -mtriple=armv6-apple-darwin -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-ARM %s
; RUN: llc < %s -filetype=obj -mtriple=armv7-apple-darwin -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-ARMV7 %s

; Empty functions need a NOP in them for MachO to prevent DWARF FDEs from
; getting all mucked up. See lib/CodeGen/AsmPrinter/AsmPrinter.cpp for
; details.
define internal fastcc void @empty_function() {
  unreachable
}
; CHECK-T1:    SectionData (
; CHECK-T1:      0000: FEDE                                 |..|
; CHECK-T1:    )
; CHECK-T2:    SectionData (
; CHECK-T2:      0000: FEDE                                 |..|
; CHECK-T2:    )
; CHECK-ARM:   SectionData (
; CHECK-ARM:     0000: FEDEFFE7                             |....|
; CHECK-ARM:   )
; CHECK-ARMV7: SectionData (
; CHECK-ARMV7:   0000: FEDEFFE7                             |....|
; CHECK-ARMV7: )
