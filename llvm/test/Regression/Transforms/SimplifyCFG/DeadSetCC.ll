; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep seteq

; Check that simplifycfg deletes a dead 'seteq' instruction when it
; folds a conditional branch into a switch instruction.

declare void %foo()
declare void %bar()

void %testcfg(uint %V) {
	%C = seteq uint %V, 18
	%D = seteq uint %V, 180
	%E = or bool %C, %D
	br bool %E, label %L1, label %Sw
Sw:
       switch uint %V, label %L1 [
              uint 15, label %L2
              uint 16, label %L2
        ]
L1:
	call void %foo()
	ret void
L2:
	call void %bar()
	ret void
}

