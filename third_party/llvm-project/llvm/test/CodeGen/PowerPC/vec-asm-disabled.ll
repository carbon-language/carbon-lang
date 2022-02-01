; RUN: not llc -mcpu=pwr7 -o /dev/null %s 2>&1 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <4 x i32> @testi1(<4 x i32> %b1, <4 x i32> %b2) #0 {
entry:
  %0 = call <4 x i32> asm "xxland $0, $1, $2", "=^wd,^wd,^wd"(<4 x i32> %b1, <4 x i32> %b2) #0
  ret <4 x i32> %0

; CHECK: error: couldn't allocate output register for constraint 'wd'
}

define signext i32 @testi2(<4 x float> %__A) #0 {
entry:
  %0 = tail call { i32, <4 x float> } asm "xxsldwi ${1:x},${2:x},${2:x},3", "=^wi,=&^wi,^wi"(<4 x float> %__A) #0
  %asmresult = extractvalue { i32, <4 x float> } %0, 0
  ret i32 %asmresult

; CHECK: error: couldn't allocate output register for constraint 'wi'
}

define float @test_ww(float %x, float %y) #0 {
  %1 = tail call float asm "xsmaxdp ${0:x},${1:x},${2:x}", "=^ww,^ww,^ww"(float %x, float %y) #0
  ret float %1
; CHECK: error: couldn't allocate output register for constraint 'ww'
}

define double @test_ws(double %x, double %y) #0 {
  %1 = tail call double asm "xsmaxdp ${0:x},${1:x},${2:x}", "=^ws,^ws,^ws"(double %x, double %y) #0
  ret double %1
; CHECK: error: couldn't allocate output register for constraint 'ws'
}

attributes #0 = { nounwind "target-features"="-vsx" }

