; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+fxsr | FileCheck %s

define void @test_fxsave(i8* %ptr) {
; CHECK-LABEL: test_fxsave
; CHECK: fxsave
  call void @llvm.x86.fxsave(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxsave(i8*)

define void @test_fxsave64(i8* %ptr) {
; CHECK-LABEL: test_fxsave64
; CHECK: fxsave64
  call void @llvm.x86.fxsave64(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxsave64(i8*)

define void @test_fxrstor(i8* %ptr) {
; CHECK-LABEL: test_fxrstor
; CHECK: fxrstor
  call void @llvm.x86.fxrstor(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor(i8*)

define void @test_fxrstor64(i8* %ptr) {
; CHECK-LABEL: test_fxrstor64
; CHECK: fxrstor64
  call void @llvm.x86.fxrstor64(i8* %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor64(i8*)
