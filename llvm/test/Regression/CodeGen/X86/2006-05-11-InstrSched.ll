; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -sched-commute-nodes &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -sched-commute-nodes -stats 2>&1 | grep 'asm-printer' | grep 39

void %foo(int* %mc, int* %bp, int* %ms, int* %xmb, int* %mpp, int* %tpmm, int* %ip, int* %tpim, int* %dpp, int* %tpdm, int* %bpi, int %M) {
entry:
	%tmp9 = setlt int %M, 5		; <bool> [#uses=1]
	br bool %tmp9, label %return, label %cond_true

cond_true:		; preds = %cond_true, %entry
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %cond_true ]		; <uint> [#uses=2]
	%tmp. = shl uint %indvar, ubyte 2		; <uint> [#uses=1]
	%tmp.10 = add uint %tmp., 1		; <uint> [#uses=2]
	%k.0.0 = cast uint %tmp.10 to int		; <int> [#uses=2]
	%tmp31 = add int %k.0.0, -1		; <int> [#uses=4]
	%tmp32 = getelementptr int* %mpp, int %tmp31		; <int*> [#uses=1]
	%tmp34 = cast int* %tmp32 to sbyte*		; <sbyte*> [#uses=1]
	%tmp = tail call <16 x sbyte> %llvm.x86.sse2.loadu.dq( sbyte* %tmp34 )		; <<16 x sbyte>> [#uses=1]
	%tmp42 = getelementptr int* %tpmm, int %tmp31		; <int*> [#uses=1]
	%tmp42 = cast int* %tmp42 to <4 x int>*		; <<4 x int>*> [#uses=1]
	%tmp46 = load <4 x int>* %tmp42		; <<4 x int>> [#uses=1]
	%tmp54 = cast <16 x sbyte> %tmp to <4 x int>		; <<4 x int>> [#uses=1]
	%tmp55 = add <4 x int> %tmp54, %tmp46		; <<4 x int>> [#uses=2]
	%tmp55 = cast <4 x int> %tmp55 to <2 x long>		; <<2 x long>> [#uses=1]
	%tmp62 = getelementptr int* %ip, int %tmp31		; <int*> [#uses=1]
	%tmp65 = cast int* %tmp62 to sbyte*		; <sbyte*> [#uses=1]
	%tmp66 = tail call <16 x sbyte> %llvm.x86.sse2.loadu.dq( sbyte* %tmp65 )		; <<16 x sbyte>> [#uses=1]
	%tmp73 = getelementptr int* %tpim, int %tmp31		; <int*> [#uses=1]
	%tmp73 = cast int* %tmp73 to <4 x int>*		; <<4 x int>*> [#uses=1]
	%tmp77 = load <4 x int>* %tmp73		; <<4 x int>> [#uses=1]
	%tmp87 = cast <16 x sbyte> %tmp66 to <4 x int>		; <<4 x int>> [#uses=1]
	%tmp88 = add <4 x int> %tmp87, %tmp77		; <<4 x int>> [#uses=2]
	%tmp88 = cast <4 x int> %tmp88 to <2 x long>		; <<2 x long>> [#uses=1]
	%tmp99 = tail call <4 x int> %llvm.x86.sse2.pcmpgt.d( <4 x int> %tmp88, <4 x int> %tmp55 )		; <<4 x int>> [#uses=1]
	%tmp99 = cast <4 x int> %tmp99 to <2 x long>		; <<2 x long>> [#uses=2]
	%tmp110 = xor <2 x long> %tmp99, < long -1, long -1 >		; <<2 x long>> [#uses=1]
	%tmp111 = and <2 x long> %tmp110, %tmp55		; <<2 x long>> [#uses=1]
	%tmp121 = and <2 x long> %tmp99, %tmp88		; <<2 x long>> [#uses=1]
	%tmp131 = or <2 x long> %tmp121, %tmp111		; <<2 x long>> [#uses=1]
	%tmp137 = getelementptr int* %mc, uint %tmp.10		; <int*> [#uses=1]
	%tmp137 = cast int* %tmp137 to <2 x long>*		; <<2 x long>*> [#uses=1]
	store <2 x long> %tmp131, <2 x long>* %tmp137
	%tmp147 = add int %k.0.0, 8		; <int> [#uses=1]
	%tmp = setgt int %tmp147, %M		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp, label %return, label %cond_true

return:		; preds = %cond_true, %entry
	ret void
}

declare <16 x sbyte> %llvm.x86.sse2.loadu.dq(sbyte*)

declare <4 x int> %llvm.x86.sse2.pcmpgt.d(<4 x int>, <4 x int>)
