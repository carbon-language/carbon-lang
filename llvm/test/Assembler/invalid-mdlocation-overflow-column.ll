; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!0 = !{}

; CHECK-NOT: error:
!1 = !MDLocation(column: 65535, scope: !0)

; CHECK: <stdin>:[[@LINE+1]]:26: error: value for 'column' too large, limit is 65535
!2 = !MDLocation(column: 65536, scope: !0)
