; RUN: llvm-as < %s | llc | grep lrvr  | count 2
; RUN: llvm-as < %s | llc | grep lrvgr | count 1
; RUN: llvm-as < %s | llc | grep lrvh  | count 1
; RUN: llvm-as < %s | llc | grep {lrv.%} | count 1
; RUN: llvm-as < %s | llc | grep {lrvg.%} | count 1


target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"


define i16 @foo(i16 zeroext %a) zeroext {
	%res = tail call i16 @llvm.bswap.i16(i16 %a)
	ret i16 %res
}

define i32 @foo2(i32 zeroext %a) zeroext {
        %res = tail call i32 @llvm.bswap.i32(i32 %a)
        ret i32 %res
}

define i64 @foo3(i64 %a) zeroext {
        %res = tail call i64 @llvm.bswap.i64(i64 %a)
        ret i64 %res
}

define i16 @foo4(i16* %b) zeroext {
	%a = load i16* %b
        %res = tail call i16 @llvm.bswap.i16(i16 %a)
        ret i16 %res
}

define i32 @foo5(i32* %b) zeroext {
	%a = load i32* %b
        %res = tail call i32 @llvm.bswap.i32(i32 %a)
        ret i32 %res
}

define i64 @foo6(i64* %b) {
	%a = load i64* %b
        %res = tail call i64 @llvm.bswap.i64(i64 %a)
        ret i64 %res
}

declare i16 @llvm.bswap.i16(i16) nounwind readnone
declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone

