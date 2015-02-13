; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:67: error: missing required field 'tag'
!0 = !MDTemplateValueParameter(scope: !{}, type: !{}, value: i32 7)
