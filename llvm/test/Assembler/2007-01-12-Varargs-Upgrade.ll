; For PR1093: This test checks that llvm-upgrade correctly translates
; the llvm.va_* intrinsics to their cannonical argument form (i8*).
; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | \
; RUN:   grep { bitcast} | wc -l | grep 5

%str = internal constant [7 x ubyte] c"%d %d\0A\00"		; <[7 x ubyte]*> [#uses=1]

implementation   ; Functions:

void %f(int %a_arg, ...) {
entry:
	%l1 = alloca sbyte*, align 4		; <sbyte**> [#uses=5]
	%l2 = alloca sbyte*, align 4		; <sbyte**> [#uses=4]
	%l3 = alloca sbyte*		; <sbyte**> [#uses=2]
	call void %llvm.va_start( sbyte** %l1 )
	call void %llvm.va_copy( sbyte** %l2, sbyte** %l3 )
	call void %llvm.va_end( sbyte** %l1 )
	call void %llvm.va_end( sbyte** %l2 )
	ret void
}

declare void %llvm.va_start(sbyte**)

declare void %llvm.va_copy(sbyte**, sbyte**)

declare int %printf(ubyte*, ...)

declare void %llvm.va_end(sbyte**)
