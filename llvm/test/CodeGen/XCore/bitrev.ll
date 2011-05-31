; RUN: llc < %s -march=xcore | FileCheck %s
declare i32 @llvm.xcore.bitrev(i32)

define i32 @bitrev(i32 %val) {
; CHECK: bitrev:
; CHECK: bitrev r0, r0
	%result = call i32 @llvm.xcore.bitrev(i32 %val)
	ret i32 %result
}
