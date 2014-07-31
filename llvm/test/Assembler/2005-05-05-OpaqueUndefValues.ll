; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

%t = type opaque
@x = global %t undef
