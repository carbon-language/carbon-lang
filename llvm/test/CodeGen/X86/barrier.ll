; RUN: llc < %s -march=x86 -mattr=-sse2 | grep lock
declare void @llvm.memory.barrier( i1 , i1 , i1 , i1 , i1)

define void @test() {
	call void @llvm.memory.barrier( i1 true, i1 true,  i1 false, i1 false, i1 false)
	ret void
}