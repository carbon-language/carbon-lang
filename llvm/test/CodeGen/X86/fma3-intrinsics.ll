; RUN: llc < %s -mtriple=x86_64-pc-win32 -mcpu=core-avx2 -mattr=avx2,+fma3 | FileCheck %s

define <4 x float> @test_x86_fmadd_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmadd132ss %xmm
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmadd132ps
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @test_x86_fmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fmadd132ps {{.*\(%r.*}}, %ymm
  %res = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) nounwind
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define <4 x float> @test_x86_fnmadd_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmadd132ss %xmm
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmadd.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmadd132ps
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @test_x86_fnmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fnmadd132ps {{.*\(%r.*}}, %ymm
  %res = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) nounwind
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone


define <4 x float> @test_x86_fmsub_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmsub132ss
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmsub.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmsub132ps
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmsub_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmsub132ss
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmsub.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmsub132ps
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

;;;;

define <2 x double> @test_x86_fmadd_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmadd132sd
  %res = call <2 x double> @llvm.x86.fma.vfmadd.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmadd.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fmadd_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmadd132pd
  %res = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmadd_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmadd132sd
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmadd.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmadd_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmadd132pd
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone



define <2 x double> @test_x86_fmsub_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmsub132sd
  %res = call <2 x double> @llvm.x86.fma.vfmsub.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmsub.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmsub132pd
  %res = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmsub_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmsub132sd
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmsub.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmsub132pd
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
