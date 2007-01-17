; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

; ModuleID = '<stdin>'
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

void %f(int %x, int %y) {
entry:
	%tmp = seteq int %x, 10		; <bool> [#uses=1]
	%tmp.not = xor bool %tmp, true		; <bool> [#uses=1]
	%tmp3 = seteq int %x, %y		; <bool> [#uses=1]
	%bothcond = and bool %tmp.not, %tmp3		; <bool> [#uses=1]
	br bool %bothcond, label %cond_true4, label %return

cond_true4:		; preds = %entry
	switch int %y, label %return [
		 int 9, label %bb
		 int 10, label %bb6
	]

bb:		; preds = %cond_true4
	call void %g( int 9 )
	ret void

bb6:		; preds = %cond_true4
	call void %g( int 10 )
	ret void

return:		; preds = %cond_true4, %entry
	ret void
}

declare void %g(int)
