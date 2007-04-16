; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | %prcontext je 1 | \
; RUN:   grep BB1_1:

%str = internal constant [14 x sbyte] c"Hello world!\0A\00"		; <[14 x sbyte]*> [#uses=1]
%str = internal constant [13 x sbyte] c"Blah world!\0A\00"		; <[13 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %main(int %argc, sbyte** %argv) {
entry:
	switch int %argc, label %UnifiedReturnBlock [
		 int 1, label %bb
		 int 2, label %bb2
	]

bb:		; preds = %entry
	%tmp1 = tail call int (sbyte*, ...)* %printf( sbyte* getelementptr ([14 x sbyte]* %str, int 0, uint 0) )		; <int> [#uses=0]
	ret int 0

bb2:		; preds = %entry
	%tmp4 = tail call int (sbyte*, ...)* %printf( sbyte* getelementptr ([13 x sbyte]* %str, int 0, uint 0) )		; <int> [#uses=0]
	ret int 0

UnifiedReturnBlock:		; preds = %entry
	ret int 0
}

declare int %printf(sbyte*, ...)
