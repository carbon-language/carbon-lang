; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5


define void @foo() {
  bitcast i32* null to i32*
  ret void
}
