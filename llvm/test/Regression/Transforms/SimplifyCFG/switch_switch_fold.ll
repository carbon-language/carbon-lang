; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep switch | wc -l | grep 1

; Test that a switch going to a switch on the same value can be merged.   All 
; three switches in this example can be merged into one big one.

declare void %foo1()
declare void %foo2()
declare void %foo3()
declare void %foo4()

void %test1(uint %V) {
        switch uint %V, label %F [
                 uint 4, label %T
                 uint 17, label %T
                 uint 5, label %T
                 uint 1234, label %F
        ]

T:
        switch uint %V, label %F [
                 uint 4, label %A
                 uint 17, label %B
		 uint 42, label %C
        ]
A:
        call void %foo1()
        ret void

B:
        call void %foo2()
        ret void
C:
	call void %foo3()
	ret void

F:
        switch uint %V, label %F [
                 uint 4, label %B
                 uint 18, label %B
		 uint 42, label %D
        ]
D:
        call void %foo4()
        ret void
}

