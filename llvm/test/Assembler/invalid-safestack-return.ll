; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: error: invalid use of function-only attribute
declare safestack void @foo()
