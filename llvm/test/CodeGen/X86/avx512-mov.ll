; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: vmovdz  %xmm0, %eax
; CHECK: ret
define i32 @test1(float %x) {
   %res = bitcast float %x to i32
   ret i32 %res
}

; CHECK-LABEL: @test2
; CHECK: vmovdz  %edi
; CHECK: ret
define <4 x i32> @test2(i32 %x) {
   %res = insertelement <4 x i32>undef, i32 %x, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test3
; CHECK: vmovqz  %rdi
; CHECK: ret
define <2 x i64> @test3(i64 %x) {
   %res = insertelement <2 x i64>undef, i64 %x, i32 0
   ret <2 x i64>%res
}

; CHECK-LABEL: @test4
; CHECK: vmovdz  (%rdi)
; CHECK: ret
define <4 x i32> @test4(i32* %x) {
   %y = load i32* %x
   %res = insertelement <4 x i32>undef, i32 %y, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test5
; CHECK: vmovssz  %xmm0, (%rdi)
; CHECK: ret
define void @test5(float %x, float* %y) {
   store float %x, float* %y, align 4
   ret void
}

; CHECK-LABEL: @test6
; CHECK: vmovsdz  %xmm0, (%rdi)
; CHECK: ret
define void @test6(double %x, double* %y) {
   store double %x, double* %y, align 8
   ret void
}

; CHECK-LABEL: @test7
; CHECK: vmovssz  (%rdi), %xmm0
; CHECK: ret
define float @test7(i32* %x) {
   %y = load i32* %x
   %res = bitcast i32 %y to float
   ret float %res
}

; CHECK-LABEL: @test8
; CHECK: vmovdz %xmm0, %eax
; CHECK: ret
define i32 @test8(<4 x i32> %x) {
   %res = extractelement <4 x i32> %x, i32 0
   ret i32 %res
}

; CHECK-LABEL: @test9
; CHECK: vmovqz %xmm0, %rax
; CHECK: ret
define i64 @test9(<2 x i64> %x) {
   %res = extractelement <2 x i64> %x, i32 0
   ret i64 %res
}
