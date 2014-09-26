; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vmovdqu8
; CHECK: ret
define <64 x i8> @test1(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <64 x i8>*
  %res = load <64 x i8>* %vaddr, align 1
  ret <64 x i8>%res
}

; CHECK-LABEL: test2
; CHECK: vmovdqu8
; CHECK: ret
define void @test2(i8 * %addr, <64 x i8> %data) {
  %vaddr = bitcast i8* %addr to <64 x i8>*
  store <64 x i8>%data, <64 x i8>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test3
; CHECK: vmovdqu8{{.*{%k[1-7]}}}
; CHECK: ret
define <64 x i8> @test3(i8 * %addr, <64 x i8> %old, <64 x i8> %mask1) {
  %mask = icmp ne <64 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <64 x i8>*
  %r = load <64 x i8>* %vaddr, align 1
  %res = select <64 x i1> %mask, <64 x i8> %r, <64 x i8> %old
  ret <64 x i8>%res
}

; CHECK-LABEL: test4
; CHECK: vmovdqu8{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <64 x i8> @test4(i8 * %addr, <64 x i8> %mask1) {
  %mask = icmp ne <64 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <64 x i8>*
  %r = load <64 x i8>* %vaddr, align 1
  %res = select <64 x i1> %mask, <64 x i8> %r, <64 x i8> zeroinitializer
  ret <64 x i8>%res
}

; CHECK-LABEL: test5
; CHECK: vmovdqu16
; CHECK: ret
define <32 x i16> @test5(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <32 x i16>*
  %res = load <32 x i16>* %vaddr, align 1
  ret <32 x i16>%res
}

; CHECK-LABEL: test6
; CHECK: vmovdqu16
; CHECK: ret
define void @test6(i8 * %addr, <32 x i16> %data) {
  %vaddr = bitcast i8* %addr to <32 x i16>*
  store <32 x i16>%data, <32 x i16>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test7
; CHECK: vmovdqu16{{.*{%k[1-7]}}}
; CHECK: ret
define <32 x i16> @test7(i8 * %addr, <32 x i16> %old, <32 x i16> %mask1) {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <32 x i16>*
  %r = load <32 x i16>* %vaddr, align 1
  %res = select <32 x i1> %mask, <32 x i16> %r, <32 x i16> %old
  ret <32 x i16>%res
}

; CHECK-LABEL: test8
; CHECK: vmovdqu16{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <32 x i16> @test8(i8 * %addr, <32 x i16> %mask1) {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <32 x i16>*
  %r = load <32 x i16>* %vaddr, align 1
  %res = select <32 x i1> %mask, <32 x i16> %r, <32 x i16> zeroinitializer
  ret <32 x i16>%res
}
