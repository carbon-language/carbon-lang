; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = !DILocalVariable(tag: DW_TAG_arg_variable, scope: !{}, arg: 65535)

; CHECK: <stdin>:[[@LINE+1]]:66: error: value for 'arg' too large, limit is 65535
!1 = !DILocalVariable(tag: DW_TAG_arg_variable, scope: !{}, arg: 65536)
