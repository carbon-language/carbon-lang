; RUN: llc -mtriple=aarch64-apple-darwin                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define i8* @test_frameaddress0() nounwind {
entry:
; CHECK-LABEL: test_frameaddress0:
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: mov x0, x29
; CHECK: ldp x29, x30, [sp], #16
; CHECK: ret
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @test_frameaddress2() nounwind {
entry:
; CHECK-LABEL: test_frameaddress2:
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: ldr x[[reg:[0-9]+]], [x29]
; CHECK: ldr x0, [x[[reg]]]
; CHECK: ldp x29, x30, [sp], #16
; CHECK: ret
  %0 = call i8* @llvm.frameaddress(i32 2)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
