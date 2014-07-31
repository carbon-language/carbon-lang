; RUN: llvm-as < %s | llvm-dis -disable-output
; RUN: verify-uselistorder < %s -preserve-bc-use-list-order

!llvm.foo = !{!0}
!0 = metadata !{i32 42}
@my.str = internal constant [4 x i8] c"foo\00"
