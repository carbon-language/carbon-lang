; RUN: llc < %s -mtriple=i686-unknown-unknown   | FileCheck %s

define void @test_fxsave(i8* %ptr) {
; CHECK-LABEL: test_fxsave
; CHECK: fxsave
  call void @llvm.x86.fxsave(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxsave(i8*)

define void @test_fxrstor(i8* %ptr) {
; CHECK-LABEL: test_fxrstor
; CHECK: fxrstor
  call void @llvm.x86.fxrstor(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor(i8*)
