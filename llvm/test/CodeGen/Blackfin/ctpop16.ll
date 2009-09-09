; RUN: llc < %s -march=bfin

declare i16 @llvm.ctpop.i16(i16)

define i16 @ctpoptest(i16 %B) {
	%b = call i16 @llvm.ctpop.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b;
}
define i16 @ctpoptest_z(i16 zeroext %B) {
	%b = call i16 @llvm.ctpop.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b;
}

define i16 @ctpoptest_s(i16 signext %B) {
	%b = call i16 @llvm.ctpop.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b;
}

