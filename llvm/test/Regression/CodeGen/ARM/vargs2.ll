; RUN: llvm-as < %s | llc -march=arm
%str = internal constant [4 x sbyte] c"%d\0A\00"		; <[4 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %f(int %a, ...) {
entry:
	%va = alloca sbyte*, align 4		; <sbyte**> [#uses=4]
	call void %llvm.va_start( sbyte** %va )
	br label %bb

bb:		; preds = %bb, %entry
	%a_addr.0 = phi int [ %a, %entry ], [ %tmp5, %bb ]		; <int> [#uses=2]
	%tmp = volatile load sbyte** %va		; <sbyte*> [#uses=2]
	%tmp2 = getelementptr sbyte* %tmp, int 4		; <sbyte*> [#uses=1]
	volatile store sbyte* %tmp2, sbyte** %va
	%tmp5 = add int %a_addr.0, -1		; <int> [#uses=1]
	%tmp = seteq int %a_addr.0, 1		; <bool> [#uses=1]
	br bool %tmp, label %bb7, label %bb

bb7:		; preds = %bb
	%tmp3 = cast sbyte* %tmp to int*		; <int*> [#uses=1]
	%tmp = load int* %tmp3		; <int> [#uses=1]
	%tmp10 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([4 x sbyte]* %str, int 0, uint 0), int %tmp )		; <int> [#uses=0]
	call void %llvm.va_end( sbyte** %va )
	ret void
}

declare void %llvm.va_start(sbyte**)

declare int %printf(sbyte*, ...)

declare void %llvm.va_end(sbyte**)
