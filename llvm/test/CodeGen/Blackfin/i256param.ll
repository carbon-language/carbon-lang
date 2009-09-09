; RUN: llc < %s -march=bfin -verify-machineinstrs
@i256_s = external global i256		; <i256*> [#uses=1]

define void @i256_ls(i256 %x) nounwind  {
	store i256 %x, i256* @i256_s
	ret void
}
