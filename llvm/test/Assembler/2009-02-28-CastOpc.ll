; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order


define void @foo() {
  bitcast i32* null to i32*
  ret void
}
