; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = !DILocalVariable(tag: DW_TAG_arg_variable, scope: !{}, arg: 1)
!1 = !DILocalVariable(tag: DW_TAG_auto_variable, scope: !{}, arg: 0)

; CHECK: <stdin>:[[@LINE+1]]:66: error: expected unsigned integer
!2 = !DILocalVariable(tag: DW_TAG_arg_variable, scope: !{}, arg: -1)
