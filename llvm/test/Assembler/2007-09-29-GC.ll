; RUN: llvm-as < %s | llvm-dis | grep "@f.*gc.*shadowstack"
; RUN: llvm-as < %s | llvm-dis | grep "@g.*gc.*java"
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

define void @f() gc "shadowstack" {
entry:
	ret void
}

define void @g() gc "java" {
entry:
	ret void
}
