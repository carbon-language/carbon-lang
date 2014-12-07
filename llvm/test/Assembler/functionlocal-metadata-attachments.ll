; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

define void @foo(i32 %v) {
entry:
; CHECK: <stdin>:[[@LINE+1]]:{{[0-9]+}}: error: unexpected function-local metadata
  ret void, !foo !{i32 %v}
}
