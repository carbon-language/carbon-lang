; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw -mattr=+avx512vl --show-mc-encoding| FileCheck %s

; CHECK-LABEL: test_256_1
; CHECK: vmovdqu8 {{.*}} ## encoding: [0x62
; CHECK: ret
define <32 x i8> @test_256_1(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <32 x i8>*
  %res = load <32 x i8>* %vaddr, align 1
  ret <32 x i8>%res
}

; CHECK-LABEL: test_256_2
; CHECK: vmovdqu8{{.*}} ## encoding: [0x62
; CHECK: ret
define void @test_256_2(i8 * %addr, <32 x i8> %data) {
  %vaddr = bitcast i8* %addr to <32 x i8>*
  store <32 x i8>%data, <32 x i8>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test_256_3
; CHECK: vmovdqu8{{.*{%k[1-7]} }}## encoding: [0x62
; CHECK: ret
define <32 x i8> @test_256_3(i8 * %addr, <32 x i8> %old, <32 x i8> %mask1) {
  %mask = icmp ne <32 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <32 x i8>*
  %r = load <32 x i8>* %vaddr, align 1
  %res = select <32 x i1> %mask, <32 x i8> %r, <32 x i8> %old
  ret <32 x i8>%res
}

; CHECK-LABEL: test_256_4
; CHECK: vmovdqu8{{.*{%k[1-7]} {z} }}## encoding: [0x62
; CHECK: ret
define <32 x i8> @test_256_4(i8 * %addr, <32 x i8> %mask1) {
  %mask = icmp ne <32 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <32 x i8>*
  %r = load <32 x i8>* %vaddr, align 1
  %res = select <32 x i1> %mask, <32 x i8> %r, <32 x i8> zeroinitializer
  ret <32 x i8>%res
}

; CHECK-LABEL: test_256_5
; CHECK: vmovdqu16{{.*}} ## encoding: [0x62
; CHECK: ret
define <16 x i16> @test_256_5(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x i16>*
  %res = load <16 x i16>* %vaddr, align 1
  ret <16 x i16>%res
}

; CHECK-LABEL: test_256_6
; CHECK: vmovdqu16{{.*}} ## encoding: [0x62
; CHECK: ret
define void @test_256_6(i8 * %addr, <16 x i16> %data) {
  %vaddr = bitcast i8* %addr to <16 x i16>*
  store <16 x i16>%data, <16 x i16>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test_256_7
; CHECK: vmovdqu16{{.*{%k[1-7]} }}## encoding: [0x62
; CHECK: ret
define <16 x i16> @test_256_7(i8 * %addr, <16 x i16> %old, <16 x i16> %mask1) {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i16>*
  %r = load <16 x i16>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i16> %r, <16 x i16> %old
  ret <16 x i16>%res
}

; CHECK-LABEL: test_256_8
; CHECK: vmovdqu16{{.*{%k[1-7]} {z} }}## encoding: [0x62
; CHECK: ret
define <16 x i16> @test_256_8(i8 * %addr, <16 x i16> %mask1) {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i16>*
  %r = load <16 x i16>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i16> %r, <16 x i16> zeroinitializer
  ret <16 x i16>%res
}

; CHECK-LABEL: test_128_1
; CHECK: vmovdqu8 {{.*}} ## encoding: [0x62
; CHECK: ret
define <16 x i8> @test_128_1(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <16 x i8>*
  %res = load <16 x i8>* %vaddr, align 1
  ret <16 x i8>%res
}

; CHECK-LABEL: test_128_2
; CHECK: vmovdqu8{{.*}} ## encoding: [0x62
; CHECK: ret
define void @test_128_2(i8 * %addr, <16 x i8> %data) {
  %vaddr = bitcast i8* %addr to <16 x i8>*
  store <16 x i8>%data, <16 x i8>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test_128_3
; CHECK: vmovdqu8{{.*{%k[1-7]} }}## encoding: [0x62
; CHECK: ret
define <16 x i8> @test_128_3(i8 * %addr, <16 x i8> %old, <16 x i8> %mask1) {
  %mask = icmp ne <16 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i8>*
  %r = load <16 x i8>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i8> %r, <16 x i8> %old
  ret <16 x i8>%res
}

; CHECK-LABEL: test_128_4
; CHECK: vmovdqu8{{.*{%k[1-7]} {z} }}## encoding: [0x62
; CHECK: ret
define <16 x i8> @test_128_4(i8 * %addr, <16 x i8> %mask1) {
  %mask = icmp ne <16 x i8> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <16 x i8>*
  %r = load <16 x i8>* %vaddr, align 1
  %res = select <16 x i1> %mask, <16 x i8> %r, <16 x i8> zeroinitializer
  ret <16 x i8>%res
}

; CHECK-LABEL: test_128_5
; CHECK: vmovdqu16{{.*}} ## encoding: [0x62
; CHECK: ret
define <8 x i16> @test_128_5(i8 * %addr) {
  %vaddr = bitcast i8* %addr to <8 x i16>*
  %res = load <8 x i16>* %vaddr, align 1
  ret <8 x i16>%res
}

; CHECK-LABEL: test_128_6
; CHECK: vmovdqu16{{.*}} ## encoding: [0x62
; CHECK: ret
define void @test_128_6(i8 * %addr, <8 x i16> %data) {
  %vaddr = bitcast i8* %addr to <8 x i16>*
  store <8 x i16>%data, <8 x i16>* %vaddr, align 1
  ret void
}

; CHECK-LABEL: test_128_7
; CHECK: vmovdqu16{{.*{%k[1-7]} }}## encoding: [0x62
; CHECK: ret
define <8 x i16> @test_128_7(i8 * %addr, <8 x i16> %old, <8 x i16> %mask1) {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i16>*
  %r = load <8 x i16>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x i16> %r, <8 x i16> %old
  ret <8 x i16>%res
}

; CHECK-LABEL: test_128_8
; CHECK: vmovdqu16{{.*{%k[1-7]} {z} }}## encoding: [0x62
; CHECK: ret
define <8 x i16> @test_128_8(i8 * %addr, <8 x i16> %mask1) {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %vaddr = bitcast i8* %addr to <8 x i16>*
  %r = load <8 x i16>* %vaddr, align 1
  %res = select <8 x i1> %mask, <8 x i16> %r, <8 x i16> zeroinitializer
  ret <8 x i16>%res
}

