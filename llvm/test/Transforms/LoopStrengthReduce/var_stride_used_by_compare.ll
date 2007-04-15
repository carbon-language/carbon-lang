; Base should not be i*3, it should be i*2.
; RUN: llvm-upgrade < %s | llvm-as | opt -loop-reduce | llvm-dis | \
; RUN:   not grep {mul.*%i, 3} 

; Indvar should not start at zero:
; RUN: llvm-upgrade < %s | llvm-as | opt -loop-reduce | llvm-dis | \
; RUN:   not grep {phi uint .* 0}
; END.

; mul uint %i, 3

; ModuleID = 't.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8"
%flags2 = external global [8193 x sbyte], align 32		; <[8193 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %foo(int %k, int %i.s) {
entry:
	%i = cast int %i.s to uint		; <uint> [#uses=2]
	%k_addr.012 = shl int %i.s, ubyte 1		; <int> [#uses=1]
	%tmp14 = setgt int %k_addr.012, 8192		; <bool> [#uses=1]
	br bool %tmp14, label %return, label %bb.preheader

bb.preheader:		; preds = %entry
	%tmp. = shl uint %i, ubyte 1		; <uint> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%indvar = phi uint [ %indvar.next, %bb ], [ 0, %bb.preheader ]		; <uint> [#uses=2]
	%tmp.15 = mul uint %indvar, %i		; <uint> [#uses=1]
	%tmp.16 = add uint %tmp.15, %tmp.		; <uint> [#uses=2]
	%k_addr.0.0 = cast uint %tmp.16 to int		; <int> [#uses=1]
	%tmp = getelementptr [8193 x sbyte]* %flags2, int 0, uint %tmp.16		; <sbyte*> [#uses=1]
	store sbyte 0, sbyte* %tmp
	%k_addr.0 = add int %k_addr.0.0, %i.s		; <int> [#uses=1]
	%tmp = setgt int %k_addr.0, 8192		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp, label %return.loopexit, label %bb

return.loopexit:		; preds = %bb
	br label %return

return:		; preds = %return.loopexit, %entry
	ret void
}
