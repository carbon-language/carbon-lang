; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub
declare int %strlen(sbyte*)
declare void %use(int %X)

sbyte %test(sbyte* %P, sbyte* %Q) {
	%A = load sbyte* %Q
	%X = call int %strlen(sbyte* %P)
	%B = load sbyte* %Q                ;; CSE with A.
	call void %use(int %X)             ;; make strlen not dead

	%C = sub sbyte %A, %B
	ret sbyte %C
}
