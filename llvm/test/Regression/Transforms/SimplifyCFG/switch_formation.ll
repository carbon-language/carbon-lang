; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep 'br'

bool %_ZN4llvm11SetCondInst7classofEPKNS_11InstructionE({uint, uint}* %I) {
entry:
	%tmp.1.i = getelementptr {uint, uint}* %I, long 0, ubyte 1
	%tmp.2.i = load uint* %tmp.1.i
	%tmp.2 = seteq uint %tmp.2.i, 14
	br bool %tmp.2, label %shortcirc_done.4, label %shortcirc_next.0

shortcirc_next.0:		; preds = %entry
	%tmp.6 = seteq uint %tmp.2.i, 15		; <bool> [#uses=1]
	br bool %tmp.6, label %shortcirc_done.4, label %shortcirc_next.1

shortcirc_next.1:		; preds = %shortcirc_next.0
	%tmp.11 = seteq uint %tmp.2.i, 16		; <bool> [#uses=1]
	br bool %tmp.11, label %shortcirc_done.4, label %shortcirc_next.2

shortcirc_next.2:		; preds = %shortcirc_next.1
	%tmp.16 = seteq uint %tmp.2.i, 17		; <bool> [#uses=1]
	br bool %tmp.16, label %shortcirc_done.4, label %shortcirc_next.3

shortcirc_next.3:		; preds = %shortcirc_next.2
	%tmp.21 = seteq uint %tmp.2.i, 18		; <bool> [#uses=1]
	br bool %tmp.21, label %shortcirc_done.4, label %shortcirc_next.4

shortcirc_next.4:		; preds = %shortcirc_next.3
	%tmp.26 = seteq uint %tmp.2.i, 19		; <bool> [#uses=1]
	br label %UnifiedReturnBlock

shortcirc_done.4:		; preds = %entry, %shortcirc_next.0, %shortcirc_next.1, %shortcirc_next.2, %shortcirc_next.3
	br label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %shortcirc_next.4, %shortcirc_done.4
	%UnifiedRetVal = phi bool [ %tmp.26, %shortcirc_next.4 ], [ true, %shortcirc_done.4 ]		; <bool> [#uses=1]
	ret bool %UnifiedRetVal
}
