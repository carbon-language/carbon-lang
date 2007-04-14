; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {ret i1 true}
; PR586

%g_07918478 = external global uint		; <uint*> [#uses=1]

implementation   ; Functions:

bool %test() {
	%tmp.0 = load uint* %g_07918478		; <uint> [#uses=2]
	%tmp.1 = setne uint %tmp.0, 0		; <bool> [#uses=1]
	%tmp.4 = setlt uint %tmp.0, 4111		; <bool> [#uses=1]
	%bothcond = or bool %tmp.1, %tmp.4		; <bool> [#uses=1]
	ret bool %bothcond
}

