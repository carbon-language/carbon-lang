; The induction variable canonicalization pass shouldn't leave dead
; instructions laying around!
;
; RUN: llvm-as < %s | opt -indvars | llvm-dis | not grep '#uses=0'

int %mul(int %x, int %y) {
entry:
	br label %tailrecurse

tailrecurse:		; preds = %entry, %endif
	%accumulator.tr = phi int [ %x, %entry ], [ %tmp.9, %endif ]		; <int> [#uses=2]
	%y.tr = phi int [ %y, %entry ], [ %tmp.8, %endif ]		; <int> [#uses=2]
	%tmp.1 = seteq int %y.tr, 0		; <bool> [#uses=1]
	br bool %tmp.1, label %return, label %endif

endif:		; preds = %tailrecurse
	%tmp.8 = add int %y.tr, -1		; <int> [#uses=1]
	%tmp.9 = add int %accumulator.tr, %x		; <int> [#uses=1]
	br label %tailrecurse

return:		; preds = %tailrecurse
	ret int %accumulator.tr
}
