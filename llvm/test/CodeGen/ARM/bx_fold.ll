; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | not grep bx

void %test(int %Ptr, ubyte* %L) {
entry:
	br label %bb1

bb:
	%tmp7 = getelementptr ubyte* %L, uint %indvar
	store ubyte 0, ubyte* %tmp7
	%indvar.next = add uint %indvar, 1
	br label %bb1

bb1:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %bb ]
	%i.0 = cast uint %indvar to int
	%Ptr_addr.0 = sub int %Ptr, %i.0
	%tmp12 = seteq int %i.0, %Ptr
	%tmp12.not = xor bool %tmp12, true
	%bothcond = and bool %tmp12.not, false
	br bool %bothcond, label %bb, label %bb18

bb18:
	ret void
}
