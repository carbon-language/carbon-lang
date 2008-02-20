; RUN: llvm-as < %s -o /dev/null -f 
; XFAIL: *

define void @test() {
	call {} @foo()
	ret void
}

declare {} @foo()
