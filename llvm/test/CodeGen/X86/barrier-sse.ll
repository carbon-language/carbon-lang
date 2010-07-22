; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep sfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep lfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep mfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep MEMBARRIER


declare void @llvm.memory.barrier( i1 , i1 , i1 , i1 , i1)

define void @test() {
	call void @llvm.memory.barrier( i1 true, i1 true,  i1 false, i1 false, i1 false)
	call void @llvm.memory.barrier( i1 true, i1 false, i1 true,  i1 false, i1 false)
	call void @llvm.memory.barrier( i1 true, i1 false, i1 false, i1 true,  i1 false)

	call void @llvm.memory.barrier( i1 true, i1 true,  i1 true,  i1 false, i1 false)
	call void @llvm.memory.barrier( i1 true, i1 true,  i1 false, i1 true,  i1 false)
	call void @llvm.memory.barrier( i1 true, i1 false, i1 true,  i1 true,  i1 false)

	call void @llvm.memory.barrier( i1 true, i1 true, i1 true, i1 true , i1 false)
	call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 false , i1 false)
	ret void
}
