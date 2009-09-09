; RUN: llc < %s -march=bfin -verify-machineinstrs

define i32 @adj(i32 %d.1, i32 %ct.1) {
entry:
	%tmp.22.not = trunc i32 %ct.1 to i1		; <i1> [#uses=1]
	%tmp.221 = xor i1 %tmp.22.not, true		; <i1> [#uses=1]
	%tmp.26 = or i1 false, %tmp.221		; <i1> [#uses=1]
	%tmp.27 = zext i1 %tmp.26 to i32		; <i32> [#uses=1]
	ret i32 %tmp.27
}
