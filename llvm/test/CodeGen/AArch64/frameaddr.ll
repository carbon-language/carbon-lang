; RUN: llc -o - %s -mtriple=arm64-apple-ios7.0  | FileCheck %s

define i8* @t() nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: mov x0, x29
	%0 = call i8* @llvm.frameaddress(i32 0)
        ret i8* %0
}

define i8* @t2() nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: ldr x[[reg:[0-9]+]], [x29]
; CHECK: ldr {{x[0-9]+}}, [x[[reg]]]
	%0 = call i8* @llvm.frameaddress(i32 2)
        ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
