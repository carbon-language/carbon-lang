; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   grep {b LBB.*} | wc -l | grep 4

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.7.0"

implementation   ; Functions:

void %foo(int %W, int %X, int %Y, int %Z) {
entry:
	%X = cast int %X to uint		; <uint> [#uses=1]
	%Y = cast int %Y to uint		; <uint> [#uses=1]
	%Z = cast int %Z to uint		; <uint> [#uses=1]
	%W = cast int %W to uint		; <uint> [#uses=1]
	%tmp1 = and int %W, 1		; <int> [#uses=1]
	%tmp1 = seteq int %tmp1, 0		; <bool> [#uses=1]
	br bool %tmp1, label %cond_false, label %bb5

bb:		; preds = %bb5, %bb
	%indvar77 = phi uint [ %indvar.next78, %bb ], [ 0, %bb5 ]		; <uint> [#uses=1]
	%tmp2 = tail call int (...)* %bar( )		; <int> [#uses=0]
	%indvar.next78 = add uint %indvar77, 1		; <uint> [#uses=2]
	%exitcond79 = seteq uint %indvar.next78, %X		; <bool> [#uses=1]
	br bool %exitcond79, label %cond_next48, label %bb

bb5:		; preds = %entry
	%tmp = seteq int %X, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_next48, label %bb

cond_false:		; preds = %entry
	%tmp10 = and int %W, 2		; <int> [#uses=1]
	%tmp10 = seteq int %tmp10, 0		; <bool> [#uses=1]
	br bool %tmp10, label %cond_false20, label %bb16

bb12:		; preds = %bb16, %bb12
	%indvar72 = phi uint [ %indvar.next73, %bb12 ], [ 0, %bb16 ]		; <uint> [#uses=1]
	%tmp13 = tail call int (...)* %bar( )		; <int> [#uses=0]
	%indvar.next73 = add uint %indvar72, 1		; <uint> [#uses=2]
	%exitcond74 = seteq uint %indvar.next73, %Y		; <bool> [#uses=1]
	br bool %exitcond74, label %cond_next48, label %bb12

bb16:		; preds = %cond_false
	%tmp18 = seteq int %Y, 0		; <bool> [#uses=1]
	br bool %tmp18, label %cond_next48, label %bb12

cond_false20:		; preds = %cond_false
	%tmp23 = and int %W, 4		; <int> [#uses=1]
	%tmp23 = seteq int %tmp23, 0		; <bool> [#uses=1]
	br bool %tmp23, label %cond_false33, label %bb29

bb25:		; preds = %bb29, %bb25
	%indvar67 = phi uint [ %indvar.next68, %bb25 ], [ 0, %bb29 ]		; <uint> [#uses=1]
	%tmp26 = tail call int (...)* %bar( )		; <int> [#uses=0]
	%indvar.next68 = add uint %indvar67, 1		; <uint> [#uses=2]
	%exitcond69 = seteq uint %indvar.next68, %Z		; <bool> [#uses=1]
	br bool %exitcond69, label %cond_next48, label %bb25

bb29:		; preds = %cond_false20
	%tmp31 = seteq int %Z, 0		; <bool> [#uses=1]
	br bool %tmp31, label %cond_next48, label %bb25

cond_false33:		; preds = %cond_false20
	%tmp36 = and int %W, 8		; <int> [#uses=1]
	%tmp36 = seteq int %tmp36, 0		; <bool> [#uses=1]
	br bool %tmp36, label %cond_next48, label %bb42

bb38:		; preds = %bb42
	%tmp39 = tail call int (...)* %bar( )		; <int> [#uses=0]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br label %bb42

bb42:		; preds = %cond_false33, %bb38
	%indvar = phi uint [ %indvar.next, %bb38 ], [ 0, %cond_false33 ]		; <uint> [#uses=3]
	%indvar = cast uint %indvar to int		; <int> [#uses=1]
	%W_addr.0 = sub int %W, %indvar		; <int> [#uses=1]
	%exitcond = seteq uint %indvar, %W		; <bool> [#uses=1]
	br bool %exitcond, label %cond_next48, label %bb38

cond_next48:		; preds = %bb, %bb12, %bb25, %bb42, %cond_false33, %bb29, %bb16, %bb5
	%W_addr.1 = phi int [ %W, %bb5 ], [ %W, %bb16 ], [ %W, %bb29 ], [ %W, %cond_false33 ], [ %W_addr.0, %bb42 ], [ %W, %bb25 ], [ %W, %bb12 ], [ %W, %bb ]		; <int> [#uses=1]
	%tmp50 = seteq int %W_addr.1, 0		; <bool> [#uses=1]
	br bool %tmp50, label %UnifiedReturnBlock, label %cond_true51

cond_true51:		; preds = %cond_next48
	%tmp52 = tail call int (...)* %bar( )		; <int> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %cond_next48
	ret void
}

declare int %bar(...)
