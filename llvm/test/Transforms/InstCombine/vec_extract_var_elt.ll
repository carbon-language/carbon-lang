; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test (float %b, <8 x float> * %p)  {
; CHECK: extractelement
; CHECK: fptosi
  %1 = load <8 x float> * %p
  %2 = bitcast <8 x float> %1 to <8 x i32>
  %3 = bitcast <8 x i32> %2 to <8 x float>
  %a = fptosi <8 x float> %3 to <8 x i32>
  %4 = fptosi float %b to i32
  %5 = add i32 %4, -2
  %6 = extractelement <8 x i32> %a, i32 %5
  %7 = insertelement <8 x i32> undef, i32 %6, i32 7
  %8 = sitofp <8 x i32> %7 to <8 x float>
  store <8 x float> %8, <8 x float>* %p
  ret void    
}

