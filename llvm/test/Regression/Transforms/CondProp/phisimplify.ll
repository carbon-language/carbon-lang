; RUN: llvm-as < %s | opt -condprop | llvm-dis | not grep phi

int %test(uint %C, bool %Val) {
        switch uint %C, label %T1 [
                 uint 4, label %T2
                 uint 17, label %T3
        ]
T1:
	call void %a()
	br label %Cont
T2:
	call void %b()
	br label %Cont
T3:
	call void %c()
	br label %Cont

Cont:
	;; PHI becomes dead after threading T2
	%C2 = phi bool [%Val, %T1], [true, %T2], [%Val, %T3]
	br bool %C2, label %L2, label %F2
L2:
	call void %d()
	ret int 17
F2:
	call void %e()
	ret int 1
}
declare void %a()
declare void %b()
declare void %c()
declare void %d()
declare void %e()
