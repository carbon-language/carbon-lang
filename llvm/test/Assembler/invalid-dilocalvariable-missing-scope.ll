; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:48: error: missing required field 'scope'
!0 = !DILocalVariable(tag: DW_TAG_auto_variable)
