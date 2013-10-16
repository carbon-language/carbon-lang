; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; Verify that llvm.ident is properly structured.
; llvm.ident takes a list of metadata entries.
; Each metadata entry can have only one string.

!llvm.ident = !{!0, !1}
!0 = metadata !{metadata !"version string"}
!1 = metadata !{metadata !"string1", metadata !"string2"}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: incorrect number of operands in llvm.ident metadata
; CHECK-NEXT: metadata !1

