; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:32: error: missing required field 'count'
!0 = !DISubrange(lowerBound: -3)
