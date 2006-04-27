; RUN: llvm-as < %s | llc
; PR748

%G = external global ushort		; <ushort*> [#uses=1]

implementation   ; Functions:

void %OmNewObjHdr() {
entry:
	br bool false, label %endif.4, label %then.0

then.0:		; preds = %entry
	ret void

endif.4:		; preds = %entry
	br bool false, label %else.3, label %shortcirc_next.3

shortcirc_next.3:		; preds = %endif.4
	ret void

else.3:		; preds = %endif.4
	switch int 0, label %endif.10 [
		 int 5001, label %then.10
		 int -5008, label %then.10
	]

then.10:		; preds = %else.3, %else.3
	%tmp.112 = load ushort* null		; <ushort> [#uses=2]
	%tmp.113 = load ushort* %G		; <ushort> [#uses=2]
	%tmp.114 = setgt ushort %tmp.112, %tmp.113		; <bool> [#uses=1]
	%tmp.120 = setlt ushort %tmp.112, %tmp.113		; <bool> [#uses=1]
	%bothcond = and bool %tmp.114, %tmp.120		; <bool> [#uses=1]
	br bool %bothcond, label %else.4, label %then.11

then.11:		; preds = %then.10
	ret void

else.4:		; preds = %then.10
	ret void

endif.10:		; preds = %else.3
	ret void
}
