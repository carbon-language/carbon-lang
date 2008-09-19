; RUN: llvm-as < %s | opt -callgraph -disable-output |& grep {Calls function}

@a = global void ()* @f		; <void ()**> [#uses=0]

define internal void @f() {
	unreachable
}
