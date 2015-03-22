; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK-NOT: error:
!0 = !GenericDebugNode(tag: 65535)

; CHECK: <stdin>:[[@LINE+1]]:29: error: value for 'tag' too large, limit is 65535
!1 = !GenericDebugNode(tag: 65536)
