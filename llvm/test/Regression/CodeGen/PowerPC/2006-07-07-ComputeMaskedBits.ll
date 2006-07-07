; RUN: llvm-as < %s | llc -mtriple=powerpc64-apple-darwin | grep extsw | wc -l | grep 2

%lens = external global ubyte*
%vals = external global int*

int %test(int %i) {
	%tmp = load ubyte** %lens
	%tmp1 = getelementptr ubyte* %tmp, int %i
	%tmp = load ubyte* %tmp1
	%tmp2 = cast ubyte %tmp to int
	%tmp3 = load int** %vals
	%tmp5 = sub int 1, %tmp2
	%tmp6 = getelementptr int* %tmp3, int %tmp5
	%tmp7 = load int* %tmp6
	ret int %tmp7
}
