; RUN: llvm-as < %s | llvm-dis | grep "quote"
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

@"a\22quote" = global i32 0
