; RUN: llvm-as < %s | opt -analyze -callgraph -disable-output | grep {Calls function 'callee'} | count 2

define internal void @callee(...) {
entry:
	unreachable
}

define void @caller() {
entry:
	call void (...)* @callee( void (...)* @callee )
	unreachable
}
