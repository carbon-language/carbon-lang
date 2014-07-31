; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

@spell_order = global [4 x i8] c"\FF\00\F7\00"

