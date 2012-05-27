; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse4a | FileCheck %s

define void @test1(i8* %p, <4 x float> %a) nounwind optsize ssp {
; CHECK: movntss
entry:
  tail call void @llvm.x86.sse4a.movnt.ss(i8* %p, <4 x float> %a) nounwind
  ret void
}

declare void @llvm.x86.sse4a.movnt.ss(i8*, <4 x float>)

define void @test2(i8* %p, <2 x double> %a) nounwind optsize ssp {
; CHECK: movntsd
entry:
  tail call void @llvm.x86.sse4a.movnt.sd(i8* %p, <2 x double> %a) nounwind
  ret void
}

declare void @llvm.x86.sse4a.movnt.sd(i8*, <2 x double>)
