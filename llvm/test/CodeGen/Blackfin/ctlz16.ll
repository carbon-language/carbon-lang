; RUN: llc < %s -march=bfin

declare i16 @llvm.ctlz.i16(i16)

define i16 @ctlztest(i16 %B) {
	%b = call i16 @llvm.ctlz.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b
}
define i16 @ctlztest_z(i16 zeroext %B) {
	%b = call i16 @llvm.ctlz.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b
}

define i16 @ctlztest_s(i16 signext %B) {
	%b = call i16 @llvm.ctlz.i16( i16 %B )		; <i16> [#uses=1]
	ret i16 %b
}

