; RUN: llc < %s -march=bfin

declare i32 @llvm.ctlz.i32(i32)
declare i32 @llvm.cttz.i32(i32)
declare i32 @llvm.ctpop.i32(i32)

define i32 @ctlztest(i32 %B) {
	%b = call i32 @llvm.ctlz.i32( i32 %B )
	ret i32 %b;
}

define i32 @cttztest(i32 %B) {
	%b = call i32 @llvm.cttz.i32( i32 %B )
	ret i32 %b;
}

define i32 @ctpoptest(i32 %B) {
	%b = call i32 @llvm.ctpop.i32( i32 %B )
	ret i32 %b;
}
