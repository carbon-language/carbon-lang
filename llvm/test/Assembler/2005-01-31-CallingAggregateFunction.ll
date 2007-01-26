; RUN: llvm-as 2>&1 < %s -o /dev/null -f | \
; RUN:    grep "LLVM functions cannot return aggregate types"

define void @test() {
	call {} @foo()
	ret void
}

declare {} @foo()
