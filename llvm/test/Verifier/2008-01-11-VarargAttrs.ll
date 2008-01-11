; RUN: not llvm-as < %s

	%struct = type {  }

declare void @foo(...)

define void @bar() {
	call void (...)* @foo(%struct* inreg null )
	ret void
}
