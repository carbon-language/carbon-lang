; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: Alias must point to function or variable

@b1 = alias i32* @c1
@c1 = alias i32* @b1
