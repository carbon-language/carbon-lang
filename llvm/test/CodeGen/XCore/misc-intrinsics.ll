; RUN: llc < %s -march=xcore | FileCheck %s
%0 = type { i32, i32 }

declare i32 @llvm.xcore.bitrev(i32)
declare i32 @llvm.xcore.crc32(i32, i32, i32)
declare %0 @llvm.xcore.crc8(i32, i32, i32)

define i32 @bitrev(i32 %val) {
; CHECK: bitrev:
; CHECK: bitrev r0, r0
	%result = call i32 @llvm.xcore.bitrev(i32 %val)
	ret i32 %result
}

define i32 @crc32(i32 %crc, i32 %data, i32 %poly) {
; CHECK: crc32:
; CHECK: crc32 r0, r1, r2
	%result = call i32 @llvm.xcore.crc32(i32 %crc, i32 %data, i32 %poly)
	ret i32 %result
}

define %0 @crc8(i32 %crc, i32 %data, i32 %poly) {
; CHECK: crc8:
; CHECK: crc8 r0, r1, r1, r2
	%result = call %0 @llvm.xcore.crc8(i32 %crc, i32 %data, i32 %poly)
	ret %0 %result
}
