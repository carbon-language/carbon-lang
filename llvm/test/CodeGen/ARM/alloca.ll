; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
void %f(uint %a) {
entry:
	%tmp = alloca sbyte, uint %a
	call void %g( sbyte* %tmp, uint %a, uint 1, uint 2, uint 3 )
	ret void
}

declare void %g(sbyte*, uint, uint, uint, uint)
