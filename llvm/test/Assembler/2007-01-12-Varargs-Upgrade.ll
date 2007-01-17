; For PR1093: This test checks that llvm-upgrade correctly translates
; the llvm.va_* intrinsics to their cannonical argument form (i8*).
; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | \
; RUN:   grep 'va_upgrade[0-9] = bitcast' | wc -l | grep 5
%str = internal constant [7 x ubyte] c"%d %d\0A\00"		; <[7 x ubyte]*> [#uses=1]

implementation   ; Functions:

void %f(int %a_arg, ...) {
entry:
	%a = cast int %a_arg to uint		; <uint> [#uses=1]
	%l1 = alloca sbyte*, align 4		; <sbyte**> [#uses=5]
	%l2 = alloca sbyte*, align 4		; <sbyte**> [#uses=4]
	%memtmp = alloca sbyte*		; <sbyte**> [#uses=2]
	call void %llvm.va_start( sbyte** %l1 )
	%tmp22 = seteq int %a_arg, 0		; <bool> [#uses=1]
	%tmp23 = volatile load sbyte** %l1		; <sbyte*> [#uses=2]
	br bool %tmp22, label %bb8, label %bb

bb:		; preds = %bb, %entry
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %bb ]		; <uint> [#uses=1]
	%tmp.0 = phi sbyte* [ %tmp23, %entry ], [ %tmp, %bb ]		; <sbyte*> [#uses=2]
	%tmp2 = getelementptr sbyte* %tmp.0, int 4		; <sbyte*> [#uses=1]
	volatile store sbyte* %tmp2, sbyte** %l1
	%tmp3 = cast sbyte* %tmp.0 to int*		; <int*> [#uses=1]
	%tmp = load int* %tmp3		; <int> [#uses=1]
	%tmp = volatile load sbyte** %l1		; <sbyte*> [#uses=2]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=2]
	%exitcond = seteq uint %indvar.next, %a		; <bool> [#uses=1]
	br bool %exitcond, label %bb8, label %bb

bb8:		; preds = %bb, %entry
	%p1.0.1 = phi int [ undef, %entry ], [ %tmp, %bb ]		; <int> [#uses=1]
	%tmp.1 = phi sbyte* [ %tmp23, %entry ], [ %tmp, %bb ]		; <sbyte*> [#uses=1]
	store sbyte* %tmp.1, sbyte** %memtmp
	call void %llvm.va_copy( sbyte** %l2, sbyte** %memtmp )
	%tmp10 = volatile load sbyte** %l2		; <sbyte*> [#uses=2]
	%tmp12 = getelementptr sbyte* %tmp10, int 4		; <sbyte*> [#uses=1]
	volatile store sbyte* %tmp12, sbyte** %l2
	%tmp13 = cast sbyte* %tmp10 to int*		; <int*> [#uses=1]
	%tmp14 = load int* %tmp13		; <int> [#uses=1]
	%tmp17 = call int (ubyte*, ...)* %printf( ubyte* getelementptr ([7 x ubyte]* %str, int 0, uint 0), int %p1.0.1, int %tmp14 )		; <int> [#uses=0]
	call void %llvm.va_end( sbyte** %l1 )
	call void %llvm.va_end( sbyte** %l2 )
	ret void
}

declare void %llvm.va_start(sbyte**)

declare void %llvm.va_copy(sbyte**, sbyte**)

declare int %printf(ubyte*, ...)

declare void %llvm.va_end(sbyte**)
