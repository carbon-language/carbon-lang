; RUN: llvm-as < %s | opt -adce -disable-output
%G = external global int*

implementation   ; Functions:

declare void %Fn(int*)

int %main(int %argc.1, sbyte** %argv.1) {
entry:		; No predecessors!
	br label %endif.42

endif.42:		; preds = %entry, %shortcirc_done.12, %then.66, %endif.42
	br bool false, label %endif.65, label %endif.42
	
then.66:		; preds = %shortcirc_done.12
	call void %Fn( int* %tmp.2846)
	br label %endif.42

endif.65:		; preds = %endif.42
	%tmp.2846 = load int** %G
	br bool false, label %shortcirc_next.12, label %shortcirc_done.12

shortcirc_next.12:		; preds = %endif.65
	br label %shortcirc_done.12

shortcirc_done.12:		; preds = %endif.65, %shortcirc_next.12
	br bool false, label %then.66, label %endif.42
}
