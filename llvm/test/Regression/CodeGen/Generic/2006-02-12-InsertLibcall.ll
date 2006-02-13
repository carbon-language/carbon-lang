; RUN: llvm-as < %s | llc
%G = external global int

void %encode_one_frame(long %tmp.2i) {
entry:
	%tmp.9 = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp.9, label %endif.0, label %shortcirc_next.0

then.5.i:		; preds = %shortcirc_next.i
	%tmp.114.i = div long %tmp.2i, 3		; <long> [#uses=1]
	%tmp.111.i = call long %lseek( int 0, long %tmp.114.i, int 1 )		; <long> [#uses=0]
	ret void

shortcirc_next.0:		; preds = %entry
	ret void

endif.0:		; preds = %entry
	%tmp.324.i = seteq int 0, 0		; <bool> [#uses=2]
	%tmp.362.i = setlt int 0, 0		; <bool> [#uses=1]
	br bool %tmp.324.i, label %else.4.i, label %then.11.i37

then.11.i37:		; preds = %endif.0
	ret void

else.4.i:		; preds = %endif.0
	br bool %tmp.362.i, label %else.5.i, label %then.12.i

then.12.i:		; preds = %else.4.i
	ret void

else.5.i:		; preds = %else.4.i
	br bool %tmp.324.i, label %then.0.i40, label %then.17.i

then.17.i:		; preds = %else.5.i
	ret void

then.0.i40:		; preds = %else.5.i
	%tmp.8.i42 = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp.8.i42, label %else.1.i56, label %then.1.i52

then.1.i52:		; preds = %then.0.i40
	ret void

else.1.i56:		; preds = %then.0.i40
	%tmp.28.i = load int* %G
	%tmp.29.i = seteq int %tmp.28.i, 1		; <bool> [#uses=1]
	br bool %tmp.29.i, label %shortcirc_next.i, label %shortcirc_done.i

shortcirc_next.i:		; preds = %else.1.i56
	%tmp.34.i = seteq int 0, 3		; <bool> [#uses=1]
	br bool %tmp.34.i, label %then.5.i, label %endif.5.i

shortcirc_done.i:		; preds = %else.1.i56
	ret void

endif.5.i:		; preds = %shortcirc_next.i
	ret void
}

declare long %lseek(int, long, int)
