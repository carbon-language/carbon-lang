; Test forward MDNode reference
; RUN: llvm-as < %s | llvm-dis -f -o /dev/null

@llvm.blah = constant metadata !{metadata !1}
!1 = metadata !{i32 23, i32 24}

