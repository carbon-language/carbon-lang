; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!0 = !{}

; CHECK-NOT: error
!1 = !MDLocation(column: 255, scope: !0)

; CHECK: <stdin>:[[@LINE+1]]:26: error: value for 'column' too large, limit is 255
!2 = !MDLocation(column: 256, scope: !0)
