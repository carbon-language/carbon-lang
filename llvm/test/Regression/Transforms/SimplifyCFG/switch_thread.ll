; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep 'call void %DEAD'

; Test that we can thread a simple known condition through switch statements.

declare void %foo1()
declare void %foo2()
declare void %DEAD()

void %test1(uint %V) {
        switch uint %V, label %A [
                 uint 4, label %T
                 uint 17, label %Done
                 uint 1234, label %A
        ]

T:  ;; V == 4 if we get here.
	call void %foo1()
        ;; This switch is always statically determined.
        switch uint %V, label %A2 [
                 uint 4, label %B
                 uint 17, label %C
		 uint 42, label %C
        ]
A2:
	call void %DEAD()
	call void %DEAD()
	%cond2 = seteq uint %V, 4    ;; always false
	br bool %cond2, label %Done, label %C

A:
	call void %foo1()
	%cond = setne uint %V, 4    ;; always true
	br bool %cond, label %Done, label %C


Done:
        ret void

B:
        call void %foo2()
	%cond3 = seteq uint %V, 4    ;; always true
	br bool %cond3, label %Done, label %C
C:
	call void %DEAD()
	ret void
}

void %test2(uint %V) {
        switch uint %V, label %A [
                 uint 4, label %T
                 uint 17, label %D
                 uint 1234, label %E
        ]

A:  ;; V != 4, 17, 1234 here.
	call void %foo1()
        ;; This switch is always statically determined.
        switch uint %V, label %E [
                 uint 4, label %C
                 uint 17, label %C
		 uint 42, label %D
        ]
C:
	call void %DEAD()  ;; unreacahble.
	ret void
T:
	call void %foo1()
	call void %foo1()
	ret void

D:
	call void %foo1()
	ret void

E:
        ret void
}

