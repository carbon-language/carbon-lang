; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:31: error: missing required field 'tag'
!0 = !MDBasicType(name: "name")
