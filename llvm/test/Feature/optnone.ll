; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Check for the presence of attribute noopt in the disassembly.

; CHECK: @foo() #0
define void @foo() #0 {
  ret void
}

; CHECK: attributes #0 = { optnone }
attributes #0 = { optnone }

