; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@foo = global i32 0
@bar = constant i32* getelementptr(i32, i32* @foo)

