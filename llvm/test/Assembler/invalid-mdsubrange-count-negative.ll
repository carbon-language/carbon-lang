; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:25: error: expected unsigned integer
!0 = !MDSubrange(count: -3)
