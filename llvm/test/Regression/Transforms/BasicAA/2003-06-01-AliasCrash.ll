; RUN: as < %s | opt -basicaa -aa-eval -disable-output

int %MTConcat([3 x int]* %a.1) {
	%tmp.961 = getelementptr [3 x int]* %a.1, long 0, long 4
	%tmp.97 = load int* %tmp.961
	%tmp.119 = getelementptr [3 x int]* %a.1, long 1, long 0
	%tmp.120 = load int* %tmp.119
	%tmp.1541 = getelementptr [3 x int]* %a.1, long 0, long 4
	%tmp.155 = load int* %tmp.1541
	ret int 0
}
