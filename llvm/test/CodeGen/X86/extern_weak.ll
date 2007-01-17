; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=i686-apple-darwin | grep weak_reference | wc -l | grep 2

%Y = global int (sbyte*)* %X
declare extern_weak int %X(sbyte*)

void %bar() {
	tail call void (...)* %foo( )
	ret void
}

declare extern_weak void %foo(...)
