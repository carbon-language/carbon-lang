; RUN: ignore llvm-as < %s -o /dev/null -f |& \
; RUN:    grep {LLVM functions cannot return aggregate types}

define void @test() {
	call {} @foo()
	ret void
}

declare {} @foo()
