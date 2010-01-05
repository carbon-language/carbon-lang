; RUN: llc < %s -march=bfin

declare i64 @llvm.ctlz.i64(i64)
declare i64 @llvm.cttz.i64(i64)
declare i64 @llvm.ctpop.i64(i64)

define i64 @ctlztest(i64 %B) {
	%b = call i64 @llvm.ctlz.i64( i64 %B )
	ret i64 %b
}

define i64 @cttztest(i64 %B) {
	%b = call i64 @llvm.cttz.i64( i64 %B )
	ret i64 %b
}

define i64 @ctpoptest(i64 %B) {
	%b = call i64 @llvm.ctpop.i64( i64 %B )
	ret i64 %b
}
