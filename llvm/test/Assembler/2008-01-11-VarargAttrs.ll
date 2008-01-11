; RUN: llvm-as < %s | llvm-dis | grep byval | count 2

	%struct = type {  }

declare void @foo(...)

define void @bar() {
	call void (...)* @foo(%struct* byval null, %struct* byval null )
	ret void
}
