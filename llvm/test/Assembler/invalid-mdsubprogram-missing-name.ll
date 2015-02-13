; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:38: error: missing required field 'name'
!0 = !MDSubprogram(linkageName: "foo")
