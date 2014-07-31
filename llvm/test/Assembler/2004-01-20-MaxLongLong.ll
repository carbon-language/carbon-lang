; RUN: llvm-as < %s | llvm-dis | grep 9223372036854775808
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

global i64 -9223372036854775808

