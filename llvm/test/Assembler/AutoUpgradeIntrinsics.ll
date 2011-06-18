; Tests to make sure intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s


declare <4 x float> @llvm.x86.sse.loadu.ps(i8*) nounwind readnone
declare <16 x i8> @llvm.x86.sse2.loadu.dq(i8*) nounwind readnone
declare <2 x double> @llvm.x86.sse2.loadu.pd(double*) nounwind readnone
define void @test_loadu(i8* %a, double* %b) {
  %v0 = call <4 x float> @llvm.x86.sse.loadu.ps(i8* %a)
  %v1 = call <16 x i8> @llvm.x86.sse2.loadu.dq(i8* %a)
  %v2 = call <2 x double> @llvm.x86.sse2.loadu.pd(double* %b)

; CHECK: load i128* {{.*}}, align 1
; CHECK: load i128* {{.*}}, align 1
; CHECK: load i128* {{.*}}, align 1
  ret void
}

declare void @llvm.x86.sse.movnt.ps(i8*, <4 x float>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.dq(i8*, <2 x double>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.pd(i8*, <2 x double>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.i(i8*, i32) nounwind readnone 

define void @f(<4 x float> %A, i8* %B, <2 x double> %C, i32 %D) {
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse.movnt.ps(i8* %B, <4 x float> %A)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.dq(i8* %B, <2 x double> %C)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.pd(i8* %B, <2 x double> %C)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.i(i8* %B, i32 %D)
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32) nounwind

define void @p(i8* %ptr) {
; CHECK: llvm.prefetch(i8* %ptr, i32 0, i32 1, i32 1)
  tail call void @llvm.prefetch(i8* %ptr, i32 0, i32 1)
  ret void
}
