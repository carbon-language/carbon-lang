; RUN: llc < %s -march=bfin -verify-machineinstrs
@i56_l = external global i56		; <i56*> [#uses=1]
@i56_s = external global i56		; <i56*> [#uses=1]

define void @i56_ls(i56 %x) nounwind  {
	store i56 %x, i56* @i56_s
	ret void
}
