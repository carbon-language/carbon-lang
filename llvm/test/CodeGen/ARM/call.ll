; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep {mov lr, pc}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5t | grep blx
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mtriple=arm-linux-gnueabi\
; RUN:   -relocation-model=pic | grep {PLT}

%t = weak global int ()* null
declare void %g(int, int, int, int)

void %f() {
	call void %g( int 1, int 2, int 3, int 4 )
	ret void
}

void %g() {
	%tmp = load int ()** %t
	%tmp = tail call int %tmp( )
	ret void
}
