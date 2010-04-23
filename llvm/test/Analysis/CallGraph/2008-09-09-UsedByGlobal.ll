; RUN: opt < %s -print-callgraph -disable-output |& grep {calls function}

@a = global void ()* @f		; <void ()**> [#uses=0]

define internal void @f() {
	unreachable
}
