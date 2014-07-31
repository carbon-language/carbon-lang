; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

%t = type opaque
@x = global %t undef
