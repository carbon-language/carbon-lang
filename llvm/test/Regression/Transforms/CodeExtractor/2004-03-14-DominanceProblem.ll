; RUN: llvm-as < %s | opt -loop-extract -disable-output
; This testcase is failing the loop extractor because not all exit blocks 
; are dominated by all of the live-outs.

implementation   ; Functions:

int %ab(int %alpha, int %beta) {
entry:
	br label %loopentry.1.preheader

loopentry.1.preheader:		; preds = %then.1
	br label %loopentry.1

loopentry.1:		; preds = %loopentry.1.preheader, %no_exit.1
	br bool false, label %no_exit.1, label %loopexit.0.loopexit1

no_exit.1:		; preds = %loopentry.1
	%tmp.53 = load int* null		; <int> [#uses=1]
	br bool false, label %shortcirc_next.2, label %loopentry.1

shortcirc_next.2:		; preds = %no_exit.1
	%tmp.563 = call int %wins( int 0, int %tmp.53, int 3 )		; <int> [#uses=0]
	ret int 0

loopexit.0.loopexit1:		; preds = %loopentry.1
	br label %loopexit.0

loopexit.0:		; preds = %loopexit.0.loopexit, %loopexit.0.loopexit1
	ret int 0
}

declare int %wins(int, int, int)

declare ushort %ab_code()
