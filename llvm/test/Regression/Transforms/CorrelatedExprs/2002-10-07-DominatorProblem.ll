; RUN: llvm-as < %s | opt -cee
;
; The 'cee' pass is breaking SSA form when it blindly forwards the branch from 
; Eq to branch to "Forwarded" instead.

implementation

int %test(int %A, int %B, bool %c0) {
Start:
	%c1 = seteq int %A, %B
	br bool %c1, label %Eq, label %Loop

Eq:	; In this block we know that A == B
	br label %Loop    ; This should be modified to branch to "Forwarded".

Loop:        ;; Merge point, nothing is known here...
	%Z = phi int [%A, %Start], [%B, %Eq], [%Z, %Bottom]
	%c2 = setgt int %A, %B
	br bool %c2, label %Forwarded, label %Bottom

Forwarded:
	%Z2 = phi int [%Z, %Loop]
	call int %test(int 0, int %Z2, bool true)
	br label %Bottom

Bottom:
	br label %Loop
}
