; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=fma4 | FileCheck %s

define < 2 x double > @test_x86_fma4_vfmadd_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddsd
  %res = call < 2 x double > @llvm.x86.fma4.vfmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma4.vfmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

