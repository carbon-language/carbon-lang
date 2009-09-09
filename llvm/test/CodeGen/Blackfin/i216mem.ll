; RUN: llc < %s -march=bfin -verify-machineinstrs
@i216_l = external global i216		; <i216*> [#uses=1]
@i216_s = external global i216		; <i216*> [#uses=1]

define void @i216_ls() nounwind  {
	%tmp = load i216* @i216_l		; <i216> [#uses=1]
	store i216 %tmp, i216* @i216_s
	ret void
}
