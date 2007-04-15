; RUN: llvm-upgrade < %s | llvm-as | opt -domtree -break-crit-edges -analyze \
; RUN:  -domtree | grep {3.*%brtrue$}
; PR932
implementation   ; Functions:

declare void %use1(int)

void %f(int %i, bool %c) {
entry:
	%A = seteq int %i, 0		; <bool> [#uses=1]
	br bool %A, label %brtrue, label %brfalse

brtrue:		; preds = %brtrue, %entry
	%B = phi bool [ true, %brtrue ], [ false, %entry ]		; <bool> [#uses=1]
	call void %use1( int %i )
	br bool %B, label %brtrue, label %brfalse

brfalse:		; preds = %brtrue, %entry
	call void %use1( int %i )
	ret void
}
