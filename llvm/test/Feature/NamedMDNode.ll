; RUN: llvm-as < %s | llvm-dis | grep "llvm.stuff = "

;; Simple NamedMDNode
!0 = metadata !{i32 42}
!1 = metadata !{metadata !"foo"}
!llvm.stuff = !{!0, !1, null}

!samename = !{!0, !1}
define void @samename() {}
