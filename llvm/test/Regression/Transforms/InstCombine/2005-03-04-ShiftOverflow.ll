; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'ret bool false'
bool %test(ulong %tmp.169) {
	%tmp.1710 = shr ulong %tmp.169, ubyte 1
	%tmp.1912 = setgt ulong %tmp.1710, 0
	ret bool %tmp.1912
}

