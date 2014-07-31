; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

@spell_order = global [4 x i8] c"\FF\00\F7\00"

