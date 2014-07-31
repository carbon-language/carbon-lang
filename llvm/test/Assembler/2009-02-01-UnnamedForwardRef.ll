; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order
; PR3372

@X = global i32* @0
global i32 4

