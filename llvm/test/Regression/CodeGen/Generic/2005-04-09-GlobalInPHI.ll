; RUN: llvm-as < %s | llc 
	%struct.TypHeader = type { uint, %struct.TypHeader**, [3 x sbyte], ubyte }
%.str_67 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=1]
%.str_87 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %PrBinop() {
entry:
	br bool false, label %cond_true, label %else.0

cond_true:		; preds = %entry
	br label %else.0

else.0:
	%tmp.167.1 = phi int [ cast ([17 x sbyte]* %.str_87 to int), %entry ], [ 0, %cond_true ]
	call void %Pr( sbyte* getelementptr ([4 x sbyte]* %.str_67, int 0, int 0), int 0, int 0 )
	ret void
}

declare void %Pr(sbyte*, int, int)
