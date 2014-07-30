; RUN: llc  -mtriple=x86_64-apple-macosx10.9.0  -mcpu=core2 -mattr=+64bit,+sse2 < %s | FileCheck %s

; DAGCombine may choose to rewrite 2 loads feeding a select as a select of
; addresses feeding a load. This test ensures that when it does that it creates
; a load with alignment equivalent to the most restrictive source load.

declare void @sink(<2 x double>)

define void @test1(i1 %cmp) align 2 {
  %1 = alloca  <2 x double>, align 16
  %2 = alloca  <2 x double>, align 8

  %val = load <2 x double>* %1, align 16
  %val2 = load <2 x double>* %2, align 8
  %val3 = select i1 %cmp, <2 x double> %val, <2 x double> %val2
  call void @sink(<2 x double> %val3)
  ret void
  ; CHECK: test1
  ; CHECK: movups
  ; CHECK: ret
}

define void @test2(i1 %cmp) align 2 {
  %1 = alloca  <2 x double>, align 16
  %2 = alloca  <2 x double>, align 8

  %val = load <2 x double>* %1, align 16
  %val2 = load <2 x double>* %2, align 16
  %val3 = select i1 %cmp, <2 x double> %val, <2 x double> %val2
  call void @sink(<2 x double> %val3)
  ret void
  ; CHECK: test2
  ; CHECK: movaps
  ; CHECK: ret
}
