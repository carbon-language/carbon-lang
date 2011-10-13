; RUN: llc < %s -march=x86-64 -mattr=+bmi | FileCheck %s

define i32 @t1(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.cttz.i32( i32 %x )
	ret i32 %tmp
; CHECK: t1:
; CHECK: tzcntl
}

declare i32 @llvm.cttz.i32(i32) nounwind readnone

define i16 @t2(i16 %x) nounwind  {
	%tmp = tail call i16 @llvm.cttz.i16( i16 %x )
	ret i16 %tmp
; CHECK: t2:
; CHECK: tzcntw
}

declare i16 @llvm.cttz.i16(i16) nounwind readnone

define i64 @t3(i64 %x) nounwind  {
	%tmp = tail call i64 @llvm.cttz.i64( i64 %x )
	ret i64 %tmp
; CHECK: t3:
; CHECK: tzcntq
}

declare i64 @llvm.cttz.i64(i64) nounwind readnone

define i8 @t4(i8 %x) nounwind  {
	%tmp = tail call i8 @llvm.cttz.i8( i8 %x )
	ret i8 %tmp
; CHECK: t4:
; CHECK: tzcntw
}

declare i8 @llvm.cttz.i8(i8) nounwind readnone

