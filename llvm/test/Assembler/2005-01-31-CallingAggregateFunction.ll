; RUN: llvm-as < %s -o /dev/null -f 

define void @test() {
	call {} @foo()
	ret void
}

declare {} @foo()
