; Test to make sure that SSA is correctly updated
; RUN: llvm-as < %s | opt -cee
;
implementation   ; Functions:

int %test(int %A, int %B, bool %c0) {
Start:		; No predecessors!
	%c1 = seteq int %A, %B		; <bool> [#uses=1]
	br bool %c1, label %Eq, label %Start_crit_edge

Start_crit_edge:		; preds = %Start
	br label %Loop

Eq:		; preds = %Start
	br label %Loop

Loop:		; preds = %Bottom, %Eq, %Start_crit_edge
	%Z = phi int [ %A, %Start_crit_edge ], [ %B, %Eq ], [ %Z, %Bottom ]		; <int> [#uses=2]
	%c2 = setge int %A, %B		; <bool> [#uses=1]
	br bool %c2, label %Forwarded, label %Loop_crit_edge

Loop_crit_edge:		; preds = %Loop
	br label %Bottom

Forwarded:		; preds = %Loop
	%Z2 = phi int [ %Z, %Loop ]		; <int> [#uses=1]
	call int %test( int 0, int %Z2, bool true )		; <int>:0 [#uses=0]
	br label %Bottom

Bottom:		; preds = %Forwarded, %Loop_crit_edge
	br label %Loop
}
