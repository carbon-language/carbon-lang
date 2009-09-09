; RUN: llc < %s -march=bfin -verify-machineinstrs
@i1_l = external global i1		; <i1*> [#uses=1]
@i1_s = external global i1		; <i1*> [#uses=1]

define void @i1_ls() nounwind  {
	%tmp = load i1* @i1_l		; <i1> [#uses=1]
	store i1 %tmp, i1* @i1_s
	ret void
}
