; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding| FileCheck %s

; CHECK-LABEL: @test1
; CHECK: vmovd  %xmm0, %eax ## encoding: [0x62
; CHECK: ret
define i32 @test1(float %x) {
   %res = bitcast float %x to i32
   ret i32 %res
}

; CHECK-LABEL: @test2
; CHECK: vmovd  %edi, %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x i32> @test2(i32 %x) {
   %res = insertelement <4 x i32>undef, i32 %x, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test3
; CHECK: vmovq  %rdi, %xmm0 ## encoding: [0x62
; CHECK: ret
define <2 x i64> @test3(i64 %x) {
   %res = insertelement <2 x i64>undef, i64 %x, i32 0
   ret <2 x i64>%res
}

; CHECK-LABEL: @test4
; CHECK: vmovd  (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x i32> @test4(i32* %x) {
   %y = load i32* %x
   %res = insertelement <4 x i32>undef, i32 %y, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test5
; CHECK: vmovss  %xmm0, (%rdi) ## encoding: [0x62
; CHECK: ret
define void @test5(float %x, float* %y) {
   store float %x, float* %y, align 4
   ret void
}

; CHECK-LABEL: @test6
; CHECK: vmovsd  %xmm0, (%rdi) ## encoding: [0x62
; CHECK: ret
define void @test6(double %x, double* %y) {
   store double %x, double* %y, align 8
   ret void
}

; CHECK-LABEL: @test7
; CHECK: vmovss  (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define float @test7(i32* %x) {
   %y = load i32* %x
   %res = bitcast i32 %y to float
   ret float %res
}

; CHECK-LABEL: @test8
; CHECK: vmovd %xmm0, %eax ## encoding: [0x62
; CHECK: ret
define i32 @test8(<4 x i32> %x) {
   %res = extractelement <4 x i32> %x, i32 0
   ret i32 %res
}

; CHECK-LABEL: @test9
; CHECK: vmovq %xmm0, %rax ## encoding: [0x62
; CHECK: ret
define i64 @test9(<2 x i64> %x) {
   %res = extractelement <2 x i64> %x, i32 0
   ret i64 %res
}

; CHECK-LABEL: @test10
; CHECK: vmovd (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x i32> @test10(i32* %x) {
   %y = load i32* %x, align 4
   %res = insertelement <4 x i32>zeroinitializer, i32 %y, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test11
; CHECK: vmovss  (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x float> @test11(float* %x) {
   %y = load float* %x, align 4
   %res = insertelement <4 x float>zeroinitializer, float %y, i32 0
   ret <4 x float>%res
}

; CHECK-LABEL: @test12
; CHECK: vmovsd  (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define <2 x double> @test12(double* %x) {
   %y = load double* %x, align 8
   %res = insertelement <2 x double>zeroinitializer, double %y, i32 0
   ret <2 x double>%res
}

; CHECK-LABEL: @test13
; CHECK: vmovq  %rdi, %xmm0 ## encoding: [0x62
; CHECK: ret
define <2 x i64> @test13(i64 %x) {
   %res = insertelement <2 x i64>zeroinitializer, i64 %x, i32 0
   ret <2 x i64>%res
}

; CHECK-LABEL: @test14
; CHECK: vmovd  %edi, %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x i32> @test14(i32 %x) {
   %res = insertelement <4 x i32>zeroinitializer, i32 %x, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: @test15
; CHECK: vmovd  (%rdi), %xmm0 ## encoding: [0x62
; CHECK: ret
define <4 x i32> @test15(i32* %x) {
   %y = load i32* %x, align 4
   %res = insertelement <4 x i32>zeroinitializer, i32 %y, i32 0
   ret <4 x i32>%res
}

; CHECK-LABEL: test16
; CHECK: vmovdqu32
; CHECK: ret
define <16 x i32> @test16(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %res = load <16 x i32>* %vaddr, align 1
  ret <16 x i32>%res
}

; CHECK-LABEL: test17
; CHECK: vmovdqa32
; CHECK: ret
define <16 x i32> @test17(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %res = load <16 x i32>* %vaddr, align 64
  ret <16 x i32>%res
}

; CHECK-LABEL: test18
; CHECK: vmovdqa64
; CHECK: ret
define void @test18(i8 * %addr, <8 x i64> %data) {
  %vaddr = bitcast i8* %addr to <8 x i64>*
  store <8 x i64>%data, <8 x i64>* %vaddr, align 64
  ret void
}

; CHECK-LABEL: store_i1
; CHECK: movb
; CHECK: movb
; CHECK: ret
define void @store_i1() {
  store i1 true, i1 addrspace(3)* undef, align 128
  store i1 false, i1 addrspace(2)* undef, align 128
  ret void
}