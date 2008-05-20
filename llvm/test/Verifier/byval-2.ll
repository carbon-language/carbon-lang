; RUN: llvm-as %s -o /dev/null -f
	%s = type opaque
declare void @h(%s* byval %num)
