; RUN: llc < %s -march=bfin

@i8_l = external global i8		; <i8*> [#uses=1]
@i8_s = external global i8		; <i8*> [#uses=1]

define void @i8_ls() nounwind  {
	%tmp = load i8* @i8_l		; <i8> [#uses=1]
	store i8 %tmp, i8* @i8_s
	ret void
}
