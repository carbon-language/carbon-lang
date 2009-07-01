; RUN: llvm-as < %s | llvm-dis > %t.ll
; RUN: grep "!0 = constant metadata !{i32 21, i32 22}" %t.ll
; RUN: grep "!1 = constant metadata !{i32 23, i32 24}" %t.ll
; RUN: grep "@llvm.blah = constant metadata !{i32 1000, i16 200, metadata !1, metadata !0}" %t.ll
!0 = constant metadata !{i32 21, i32 22}
!1 = constant metadata !{i32 23, i32 24}
@llvm.blah = constant metadata !{i32 1000, i16 200, metadata !1, metadata !0}
