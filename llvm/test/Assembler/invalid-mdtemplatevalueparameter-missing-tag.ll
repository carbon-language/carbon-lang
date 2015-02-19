; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:55: error: missing required field 'tag'
!0 = !MDTemplateValueParameter(type: !{}, value: i32 7)
