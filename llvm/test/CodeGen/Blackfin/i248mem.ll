; RUN: llc < %s -march=bfin
@i248_l = external global i248		; <i248*> [#uses=1]
@i248_s = external global i248		; <i248*> [#uses=1]

define void @i248_ls() nounwind  {
	%tmp = load i248* @i248_l		; <i248> [#uses=1]
	store i248 %tmp, i248* @i248_s
	ret void
}
