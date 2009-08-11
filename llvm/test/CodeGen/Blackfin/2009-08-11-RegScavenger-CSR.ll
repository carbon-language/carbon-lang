; RUN: llvm-as < %s | llc -march=bfin -verify-machineinstrs

declare i64 @llvm.cttz.i64(i64) nounwind readnone

declare i16 @llvm.cttz.i16(i16) nounwind readnone

declare i8 @llvm.cttz.i8(i8) nounwind readnone

define void @cttztest(i8 %A, i16 %B, i32 %C, i64 %D, i8* %AP, i16* %BP, i32* %CP, i64* %DP) {
	%a = call i8 @llvm.cttz.i8(i8 %A)		; <i8> [#uses=1]
	%b = call i16 @llvm.cttz.i16(i16 %B)		; <i16> [#uses=1]
	%d = call i64 @llvm.cttz.i64(i64 %D)		; <i64> [#uses=1]
	store i8 %a, i8* %AP
	store i16 %b, i16* %BP
	store i64 %d, i64* %DP
	ret void
}
