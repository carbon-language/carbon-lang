; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @test1(float* %A, float* %C) #0 {
  %tmp1 = bitcast float* %A to <8 x float>*
  %tmp2 = load <8 x float>* %tmp1, align 32
  %tmp3 = bitcast <8 x float> %tmp2 to <8 x i32>
  %tmp4 = and <8 x i32> %tmp3, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %tmp5 = bitcast <8 x i32> %tmp4 to <8 x float>
  %tmp6 = extractelement <8 x float> %tmp5, i32 0
  store float %tmp6, float* %C
  ret void

  ; CHECK: vandps LCPI0_0(%rip), %ymm0, %ymm0
}

; Function Attrs: nounwind ssp uwtable
define void @test2(float* %A, float* %C) #0 {
  %tmp1 = bitcast float* %A to <8 x float>*
  %tmp2 = load <8 x float>* %tmp1, align 32
  %tmp3 = bitcast <8 x float> %tmp2 to <8 x i32>
  %tmp4 = or <8 x i32> %tmp3, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %tmp5 = bitcast <8 x i32> %tmp4 to <8 x float>
  %tmp6 = extractelement <8 x float> %tmp5, i32 0
  store float %tmp6, float* %C
  ret void

  ; CHECK: vorps LCPI1_0(%rip), %ymm0, %ymm0
}

; Function Attrs: nounwind ssp uwtable
define void @test3(float* %A, float* %C) #0 {
  %tmp1 = bitcast float* %A to <8 x float>*
  %tmp2 = load <8 x float>* %tmp1, align 32
  %tmp3 = bitcast <8 x float> %tmp2 to <8 x i32>
  %tmp4 = xor <8 x i32> %tmp3, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %tmp5 = bitcast <8 x i32> %tmp4 to <8 x float>
  %tmp6 = extractelement <8 x float> %tmp5, i32 0
  store float %tmp6, float* %C
  ret void

  ; CHECK: vxorps LCPI2_0(%rip), %ymm0, %ymm0
}

define void @test4(float* %A, float* %C) #0 {
  %tmp1 = bitcast float* %A to <8 x float>*
  %tmp2 = load <8 x float>* %tmp1, align 32
  %tmp3 = bitcast <8 x float> %tmp2 to <8 x i32>
  %tmp4 = xor <8 x i32> %tmp3, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %tmp5 = and <8 x i32> %tmp4, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %tmp6 = bitcast <8 x i32> %tmp5 to <8 x float>
  %tmp7 = extractelement <8 x float> %tmp6, i32 0
  store float %tmp7, float * %C
  ret void

  ;CHECK: vandnps LCPI3_0(%rip), %ymm0, %ymm0
}
