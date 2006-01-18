; The global symbol should be legalized
; RUN: llvm-as < %s | llc -march=alpha 

target endian = little
target pointersize = 64
	%struct.LIST_HELP = type { %struct.LIST_HELP*, sbyte* }
	%struct._IO_FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct._IO_FILE*, int, int, long, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, int, [44 x sbyte] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, int }
%clause_SORT = external global [21 x %struct.LIST_HELP*]		; <[21 x %struct.LIST_HELP*]*> [#uses=1]
%ia_in = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=1]
%multvec_j = external global [100 x uint]		; <[100 x uint]*> [#uses=1]

implementation   ; Functions:

void %main(int %argc) {
clock_Init.exit:
	%tmp.5.i575 = load int* null		; <int> [#uses=1]
	%tmp.309 = seteq int %tmp.5.i575, 0		; <bool> [#uses=1]
	br bool %tmp.309, label %UnifiedReturnBlock, label %then.17

then.17:		; preds = %clock_Init.exit
	store %struct._IO_FILE* null, %struct._IO_FILE** %ia_in
	%savedstack = call sbyte* %llvm.stacksave( )		; <sbyte*> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %clock_Init.exit
	ret void
}

declare sbyte* %llvm.stacksave()
