; RUN: not llvm-as %s -o /dev/null

	%struct = type {  }

declare void @foo(...)

define void @bar() {
	call void (...)* @foo(%struct* sret null )
	ret void
}
