; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32
; RUN: llvm-upgrade < %s | llvm-as | llc 

void %bitap() {
entry:
	%RMask.i = alloca [256 x uint], align 16		; <[256 x uint]*> [#uses=1]
	%buffer = alloca [147456 x sbyte], align 16		; <[147456 x sbyte]*> [#uses=0]
	br bool false, label %bb19, label %bb.preheader

bb.preheader:		; preds = %entry
	ret void

bb19:		; preds = %entry
	br bool false, label %bb12.i, label %cond_next39

bb12.i:		; preds = %bb12.i, %bb19
	%i.0.i = phi uint [ %tmp11.i, %bb12.i ], [ 0, %bb19 ]		; <uint> [#uses=2]
	%tmp9.i = getelementptr [256 x uint]* %RMask.i, int 0, uint %i.0.i		; <uint*> [#uses=1]
	store uint 0, uint* %tmp9.i
	%tmp11.i = add uint %i.0.i, 1		; <uint> [#uses=1]
	br label %bb12.i

cond_next39:		; preds = %bb19
	ret void
}
