; RUN: llvm-as < %s | llvm2cpp -funcname=WAKKA | not grep makeLLVMModule
; PR1515

define void @foo() {
  ret void
}

