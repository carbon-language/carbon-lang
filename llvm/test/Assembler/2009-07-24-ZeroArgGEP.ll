; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

@foo = global i32 0
@bar = constant i32* getelementptr(i32* @foo)

