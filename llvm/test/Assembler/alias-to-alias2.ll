; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: Alias is pointed by alias b1

@g = global i32 42

@b1 = alias i32* @c1
@c1 = alias i32* @g
