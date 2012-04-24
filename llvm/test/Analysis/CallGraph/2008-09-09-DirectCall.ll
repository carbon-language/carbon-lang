; RUN: opt < %s -print-callgraph -disable-output |& FileCheck %s

; CHECK: Call graph node <<null function>>
; CHECK:  CS<{{.*}}> calls function 'callee'
; CHECK: Call graph node for function: 'caller'
; CHECK:  CS<{{.*}}> calls function 'callee'

define internal void @callee(...) {
entry:
	unreachable
}

define void @caller() {
entry:
	call void (...)* @callee( void (...)* @callee )
	unreachable
}
