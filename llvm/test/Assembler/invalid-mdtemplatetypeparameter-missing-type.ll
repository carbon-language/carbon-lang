; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:41: error: missing required field 'type'
!0 = !MDTemplateTypeParameter(scope: !{})
