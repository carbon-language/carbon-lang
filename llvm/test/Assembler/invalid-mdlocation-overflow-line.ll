; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!0 = !{}

; CHECK-NOT: error
!1 = !MDLocation(line: 16777215, scope: !0)

; CHECK: <stdin>:[[@LINE+1]]:24: error: value for 'line' too large, limit is 16777215
!2 = !MDLocation(line: 16777216, scope: !0)
