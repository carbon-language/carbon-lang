; RUN: llvm-as <%s -bitcode-mdindex-threshold=0 | llvm-bcanalyzer -dump | FileCheck %s
; Check that distinct nodes are emitted before uniqued nodes, even if that
; breaks post-order traversals.

; Nodes in this testcase are numbered to match how they are referenced in
; bitcode.  !1 is referenced as opN=1.

; CHECK:       <DISTINCT_NODE op0=2/>
!1 = distinct !{!2}

; CHECK-NEXT:  <NODE op0=1/>
!2 = !{!1}

; Before the named records we emit the index containing the position of the
; previously emitted records
; CHECK-NEXT:   <INDEX {{.*}} (offset match)

; Note: named metadata nodes are not cannot reference null so their operands
; are numbered off-by-one.
; CHECK-NEXT:  <NAME
; CHECK-NEXT:  <NAMED_NODE op0=1/>
!named = !{!2}
