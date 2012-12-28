; RUN: opt < %s -instsimplify -S | FileCheck %s

declare {i8, i1} @llvm.uadd.with.overflow.i8(i8 %a, i8 %b)

define i1 @test_uadd1() {
; CHECK: @test_uadd1
  %x = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 254, i8 3)
  %overflow = extractvalue {i8, i1} %x, 1
  ret i1 %overflow
; CHECK-NEXT: ret i1 true
}

define i8 @test_uadd2() {
; CHECK: @test_uadd2
  %x = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 254, i8 44)
  %result = extractvalue {i8, i1} %x, 0
  ret i8 %result
; CHECK-NEXT: ret i8 42
}

declare i256 @llvm.cttz.i256(i256 %src, i1 %is_zero_undef)

define i256 @test_cttz() {
; CHECK: @test_cttz
  %x = call i256 @llvm.cttz.i256(i256 10, i1 false)
  ret i256 %x
; CHECK-NEXT: ret i256 1
}

declare i256 @llvm.ctpop.i256(i256 %src)

define i256 @test_ctpop() {
; CHECK: @test_ctpop
  %x = call i256 @llvm.ctpop.i256(i256 10)
  ret i256 %x
; CHECK-NEXT: ret i256 2
}

; Test a non-intrinsic that we know about as a library call.
declare float @fabs(float %x)

define float @test_fabs_libcall() {
; CHECK: @test_fabs_libcall

  %x = call float @fabs(float -42.0)
; This is still a real function call, so instsimplify won't nuke it -- other
; passes have to do that.
; CHECK-NEXT: call float @fabs

  ret float %x
; CHECK-NEXT: ret float 4.2{{0+}}e+01
}
