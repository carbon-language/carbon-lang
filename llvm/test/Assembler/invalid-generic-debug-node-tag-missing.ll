; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:47: error: missing required field 'tag'
!0 = !GenericDebugNode(header: "some\00header")
