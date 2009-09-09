; RUN: llc < %s -march=bfin -verify-machineinstrs
@i17_l = external global i17		; <i17*> [#uses=1]
@i17_s = external global i17		; <i17*> [#uses=1]

define void @i17_ls() nounwind  {
	%tmp = load i17* @i17_l		; <i17> [#uses=1]
	store i17 %tmp, i17* @i17_s
	ret void
}
