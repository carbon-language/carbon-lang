; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

define void @test() {
	call {i32} @foo()
	ret void
}

declare {i32 } @foo()
