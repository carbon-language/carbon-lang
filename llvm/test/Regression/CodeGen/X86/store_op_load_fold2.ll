; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep 'and DWORD PTR' | wc -l | grep 2

	%struct.Macroblock = type { int, int, int, int, int, [8 x int], %struct.Macroblock*, %struct.Macroblock*, int, [2 x [4 x [4 x [2 x int]]]], [16 x sbyte], [16 x sbyte], int, long, [4 x int], [4 x int], long, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, short, double, int, int, int, int, int, int, int, int, int }

implementation   ; Functions:

internal fastcc int %dct_chroma(int %uv, int %cr_cbp) {
entry:
	br bool false, label %bb2611, label %cond_true129

cond_true129:		; preds = %entry
	ret int 0

bb2611:		; preds = %entry
	br bool false, label %cond_true2732.preheader, label %cond_next2752

cond_true2732.preheader:		; preds = %bb2611
	%tmp2666 = getelementptr %struct.Macroblock* null, int 0, uint 13		; <long*> [#uses=2]
	%tmp2674 = cast int 0 to ubyte		; <ubyte> [#uses=1]
	br bool false, label %cond_true2732.preheader.split.us, label %cond_true2732.preheader.split

cond_true2732.preheader.split.us:		; preds = %cond_true2732.preheader
	br bool false, label %cond_true2732.outer.us.us, label %cond_true2732.outer.us

cond_true2732.outer.us.us:		; preds = %cond_true2732.preheader.split.us
	%tmp2667.us.us = load long* %tmp2666		; <long> [#uses=1]
	%tmp2670.us.us = load long* null		; <long> [#uses=1]
	%tmp2675.us.us = shl long %tmp2670.us.us, ubyte %tmp2674		; <long> [#uses=1]
	%tmp2675not.us.us = xor long %tmp2675.us.us, -1		; <long> [#uses=1]
	%tmp2676.us.us = and long %tmp2667.us.us, %tmp2675not.us.us		; <long> [#uses=1]
	store long %tmp2676.us.us, long* %tmp2666
	ret int 0

cond_true2732.outer.us:		; preds = %cond_true2732.preheader.split.us
	ret int 0

cond_true2732.preheader.split:		; preds = %cond_true2732.preheader
	ret int 0

cond_next2752:		; preds = %bb2611
	ret int 0
}
