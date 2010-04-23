; RUN: opt < %s -print-callgraph -disable-output |& \
; RUN:   grep {calls function 'callee'} | count 2

define internal void @callee(...) {
entry:
	unreachable
}

define void @caller() {
entry:
	call void (...)* @callee( void (...)* @callee )
	unreachable
}
