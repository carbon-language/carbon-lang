; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm

void %frame_dummy() {
entry:
	%tmp1 = tail call void (sbyte*)* (void (sbyte*)*)* asm "", "=r,0,~{dirflag},~{fpsr},~{flags}"( void (sbyte*)* null )
	ret void
}
