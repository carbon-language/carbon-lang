; RUN: llvm-as < %s | llvm-dis | llvm-as -f -o /dev/null
!0 = constant metadata !{i32 21, i32 22}
@llvm.blah = constant metadata !{i32 1000, i16 200, metadata !0, metadata !0}
