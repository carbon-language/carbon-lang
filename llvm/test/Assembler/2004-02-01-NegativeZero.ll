; RUN: llvm-as < %s | llvm-dis | grep -- -0.0
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

global double 0x8000000000000000
global float -0.0

