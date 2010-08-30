; RUN: llc < %s | FileCheck %s


target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-ibm-linux"


define i16 @foo(i16 zeroext %a) zeroext {
	%res = tail call i16 @llvm.bswap.i16(i16 %a)
	ret i16 %res
}

define i32 @foo2(i32 zeroext %a) zeroext {
; CHECK: foo2:
; CHECK:  lrvr %r1, %r2
        %res = tail call i32 @llvm.bswap.i32(i32 %a)
        ret i32 %res
}

define i64 @foo3(i64 %a) zeroext {
; CHECK: foo3:
; CHECK:  lrvgr %r2, %r2
        %res = tail call i64 @llvm.bswap.i64(i64 %a)
        ret i64 %res
}

define i16 @foo4(i16* %b) zeroext {
	%a = load i16* %b
        %res = tail call i16 @llvm.bswap.i16(i16 %a)
        ret i16 %res
}

define i32 @foo5(i32* %b) zeroext {
; CHECK: foo5:
; CHECK:  lrv %r1, 0(%r2)
	%a = load i32* %b
        %res = tail call i32 @llvm.bswap.i32(i32 %a)
        ret i32 %res
}

define i64 @foo6(i64* %b) {
; CHECK: foo6:
; CHECK:  lrvg %r2, 0(%r2)
	%a = load i64* %b
        %res = tail call i64 @llvm.bswap.i64(i64 %a)
        ret i64 %res
}

define void @foo7(i16 %a, i16* %b) {
        %res = tail call i16 @llvm.bswap.i16(i16 %a)
        store i16 %res, i16* %b
        ret void
}

define void @foo8(i32 %a, i32* %b) {
; CHECK: foo8:
; CHECK:  strv %r2, 0(%r3)
        %res = tail call i32 @llvm.bswap.i32(i32 %a)
        store i32 %res, i32* %b
        ret void
}

define void @foo9(i64 %a, i64* %b) {
; CHECK: foo9:
; CHECK:  strvg %r2, 0(%r3)
        %res = tail call i64 @llvm.bswap.i64(i64 %a)
        store i64 %res, i64* %b
        ret void
}

declare i16 @llvm.bswap.i16(i16) nounwind readnone
declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone

