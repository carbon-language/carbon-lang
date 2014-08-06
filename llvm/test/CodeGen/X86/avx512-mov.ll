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

; CHECK-LABEL: test19
; CHECK: vmovdqu32
; CHECK: ret
define void @test19(i8 * %addr, <16 x i32> %data) {
  %vaddr = bitcast i8* %addr to <16 x i32>*
  store <16 x i32>%data, <16 x i32>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test20
; CHECK: vmovdqa32
; CHECK: ret
define void @test20(i8 * %addr, <16 x i32> %data) {
  %vaddr = bitcast i8* %addr to <16 x i32>*
  store <16 x i32>%data, <16 x i32>* %vaddr, align 64
  ret void
}

; CHECK-LABEL: test21
; CHECK: vmovdqa64
; CHECK: ret
define  <8 x i64> @test21(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %res = load <8 x i64>* %vaddr, align 64
  ret <8 x i64>%res
}

; CHECK-LABEL: test22
; CHECK: vmovdqu64
; CHECK: ret
define void @test22(i8 * %addr, <8 x i64> %data) {
  %vaddr = bitcast i8* %addr to <8 x i64>*
  store <8 x i64>%data, <8 x i64>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test23
; CHECK: vmovdqu64
; CHECK: ret
define <8 x i64> @test23(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %res = load <8 x i64>* %vaddr, align 1
  ret <8 x i64>%res
}

; CHECK-LABEL: test24
; CHECK: vmovapd
; CHECK: ret
define void @test24(i8 * %addr, <8 x double> %data) {
  %vaddr = bitcast i8* %addr to <8 x double>*
  store <8 x double>%data, <8 x double>* %vaddr, align 64
  ret void
}

; CHECK-LABEL: test25
; CHECK: vmovapd
; CHECK: ret
define <8 x double> @test25(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <8 x double>*
  %res = load <8 x double>* %vaddr, align 64
  ret <8 x double>%res
}

; CHECK-LABEL: test26
; CHECK: vmovaps
; CHECK: ret
define void @test26(i8 * %addr, <16 x float> %data) {
  %vaddr = bitcast i8* %addr to <16 x float>*
  store <16 x float>%data, <16 x float>* %vaddr, align 64
  ret void
}

; CHECK-LABEL: test27
; CHECK: vmovaps
; CHECK: ret
define <16 x float> @test27(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x float>*
  %res = load <16 x float>* %vaddr, align 64
  ret <16 x float>%res
}

; CHECK-LABEL: test28
; CHECK: vmovupd
; CHECK: ret
define void @test28(i8 * %addr, <8 x double> %data) {
  %vaddr = bitcast i8* %addr to <8 x double>*
  store <8 x double>%data, <8 x double>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test29
; CHECK: vmovupd
; CHECK: ret
define <8 x double> @test29(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <8 x double>*
  %res = load <8 x double>* %vaddr, align 1
  ret <8 x double>%res
}

; CHECK-LABEL: test30
; CHECK: vmovups
; CHECK: ret
define void @test30(i8 * %addr, <16 x float> %data) {
  %vaddr = bitcast i8* %addr to <16 x float>*
  store <16 x float>%data, <16 x float>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test31
; CHECK: vmovups
; CHECK: ret
define <16 x float> @test31(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x float>*
  %res = load <16 x float>* %vaddr, align 1
  ret <16 x float>%res
}

; CHECK-LABEL: test32
; CHECK: vmovdqa32{{.*{%k[1-7]} }}
; CHECK: ret
define <16 x i32> @test32(i8 * %addr, <16 x i32> %old, <16 x i32> %mask1) {
  %mask = icmp ne <16 x i32> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %r = load <16 x i32>* %vaddr, align 64
  %res = select <16 x i1> %mask, <16 x i32> %r, <16 x i32> %old
  ret <16 x i32>%res
}

; CHECK-LABEL: test33
; CHECK: vmovdqu32{{.*{%k[1-7]} }}
; CHECK: ret
define <16 x i32> @test33(i8 * %addr, <16 x i32> %old, <16 x i32> %mask1) {
  %mask = icmp ne <16 x i32> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %r = load <16 x i32>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i32> %r, <16 x i32> %old
  ret <16 x i32>%res
}

; CHECK-LABEL: test34
; CHECK: vmovdqa32{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <16 x i32> @test34(i8 * %addr, <16 x i32> %mask1) {
  %mask = icmp ne <16 x i32> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %r = load <16 x i32>* %vaddr, align 64
  %res = select <16 x i1> %mask, <16 x i32> %r, <16 x i32> zeroinitializer
  ret <16 x i32>%res
}

; CHECK-LABEL: test35
; CHECK: vmovdqu32{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <16 x i32> @test35(i8 * %addr, <16 x i32> %mask1) {
  %mask = icmp ne <16 x i32> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i32>*
  %r = load <16 x i32>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i32> %r, <16 x i32> zeroinitializer
  ret <16 x i32>%res
}

; CHECK-LABEL: test36
; CHECK: vmovdqa64{{.*{%k[1-7]} }}
; CHECK: ret
define <8 x i64> @test36(i8 * %addr, <8 x i64> %old, <8 x i64> %mask1) {
  %mask = icmp ne <8 x i64> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %r = load <8 x i64>* %vaddr, align 64
  %res = select <8 x i1> %mask, <8 x i64> %r, <8 x i64> %old
  ret <8 x i64>%res
}

; CHECK-LABEL: test37
; CHECK: vmovdqu64{{.*{%k[1-7]} }}
; CHECK: ret
define <8 x i64> @test37(i8 * %addr, <8 x i64> %old, <8 x i64> %mask1) {
  %mask = icmp ne <8 x i64> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %r = load <8 x i64>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x i64> %r, <8 x i64> %old
  ret <8 x i64>%res
}

; CHECK-LABEL: test38
; CHECK: vmovdqa64{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <8 x i64> @test38(i8 * %addr, <8 x i64> %mask1) {
  %mask = icmp ne <8 x i64> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %r = load <8 x i64>* %vaddr, align 64
  %res = select <8 x i1> %mask, <8 x i64> %r, <8 x i64> zeroinitializer
  ret <8 x i64>%res
}

; CHECK-LABEL: test39
; CHECK: vmovdqu64{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <8 x i64> @test39(i8 * %addr, <8 x i64> %mask1) {
  %mask = icmp ne <8 x i64> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i64>*
  %r = load <8 x i64>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x i64> %r, <8 x i64> zeroinitializer
  ret <8 x i64>%res
}

; CHECK-LABEL: test40
; CHECK: vmovaps{{.*{%k[1-7]} }}
; CHECK: ret
define <16 x float> @test40(i8 * %addr, <16 x float> %old, <16 x float> %mask1) {
  %mask = fcmp one <16 x float> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x float>*
  %r = load <16 x float>* %vaddr, align 64
  %res = select <16 x i1> %mask, <16 x float> %r, <16 x float> %old
  ret <16 x float>%res
}

; CHECK-LABEL: test41
; CHECK: vmovups{{.*{%k[1-7]} }}
; CHECK: ret
define <16 x float> @test41(i8 * %addr, <16 x float> %old, <16 x float> %mask1) {
  %mask = fcmp one <16 x float> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x float>*
  %r = load <16 x float>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x float> %r, <16 x float> %old
  ret <16 x float>%res
}

; CHECK-LABEL: test42
; CHECK: vmovaps{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <16 x float> @test42(i8 * %addr, <16 x float> %mask1) {
  %mask = fcmp one <16 x float> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x float>*
  %r = load <16 x float>* %vaddr, align 64
  %res = select <16 x i1> %mask, <16 x float> %r, <16 x float> zeroinitializer
  ret <16 x float>%res
}

; CHECK-LABEL: test43
; CHECK: vmovups{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <16 x float> @test43(i8 * %addr, <16 x float> %mask1) {
  %mask = fcmp one <16 x float> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x float>*
  %r = load <16 x float>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x float> %r, <16 x float> zeroinitializer
  ret <16 x float>%res
}

; CHECK-LABEL: test44
; CHECK: vmovapd{{.*{%k[1-7]} }}
; CHECK: ret
define <8 x double> @test44(i8 * %addr, <8 x double> %old, <8 x double> %mask1) {
  %mask = fcmp one <8 x double> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x double>*
  %r = load <8 x double>* %vaddr, align 64
  %res = select <8 x i1> %mask, <8 x double> %r, <8 x double> %old
  ret <8 x double>%res
}

; CHECK-LABEL: test45
; CHECK: vmovupd{{.*{%k[1-7]} }}
; CHECK: ret
define <8 x double> @test45(i8 * %addr, <8 x double> %old, <8 x double> %mask1) {
  %mask = fcmp one <8 x double> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x double>*
  %r = load <8 x double>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x double> %r, <8 x double> %old
  ret <8 x double>%res
}

; CHECK-LABEL: test46
; CHECK: vmovapd{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <8 x double> @test46(i8 * %addr, <8 x double> %mask1) {
  %mask = fcmp one <8 x double> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x double>*
  %r = load <8 x double>* %vaddr, align 64
  %res = select <8 x i1> %mask, <8 x double> %r, <8 x double> zeroinitializer
  ret <8 x double>%res
}

; CHECK-LABEL: test47
; CHECK: vmovupd{{.*{%k[1-7]} {z} }}
; CHECK: ret
define <8 x double> @test47(i8 * %addr, <8 x double> %mask1) {
  %mask = fcmp one <8 x double> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x double>*
  %r = load <8 x double>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x double> %r, <8 x double> zeroinitializer
  ret <8 x double>%res
}
