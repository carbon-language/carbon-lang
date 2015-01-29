; RUN: llc < %s -march=x86 -mattr=-sse2,-sse4a | FileCheck %s

define void @test() {
; CHECK: lock
	fence seq_cst
	ret void
}
