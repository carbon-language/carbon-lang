; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:33: error: 'scope' cannot be null
!0 = !MDLexicalBlockFile(scope: null)
