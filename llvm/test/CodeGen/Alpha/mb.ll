; RUN: llc < %s -march=alpha | grep mb

define void @test() {
	fence seq_cst
	ret void
}
