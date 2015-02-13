; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:29: error: missing required field 'tag'
!0 = !MDLocalVariable(arg: 7)
