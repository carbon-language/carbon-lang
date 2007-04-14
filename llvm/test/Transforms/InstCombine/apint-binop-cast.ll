; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast

define i47 @testAdd(i31 %X, i31 %Y) {
	%tmp = add i31 %X, %Y
	%tmp.l = sext i31 %tmp to i47
	ret i47 %tmp.l
}

define i747 @testAdd2(i131 %X, i131 %Y) {
	%tmp = add i131 %X, %Y
	%tmp.l = sext i131 %tmp to i747
	ret i747 %tmp.l
}
