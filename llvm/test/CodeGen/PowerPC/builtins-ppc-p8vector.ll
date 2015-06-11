; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+power8-vector -mattr=-vsx < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-VSX

@vsc = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5>, align 16
@vuc = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5>, align 16
@res_vll = common global <2 x i64> zeroinitializer, align 16
@res_vull = common global <2 x i64> zeroinitializer, align 16
@res_vsc = common global <16 x i8> zeroinitializer, align 16
@res_vuc = common global <16 x i8> zeroinitializer, align 16

; Function Attrs: nounwind
define void @test1() {
entry:
  %__a.addr.i = alloca <16 x i8>, align 16
  %__b.addr.i = alloca <16 x i8>, align 16
  %0 = load <16 x i8>, <16 x i8>* @vsc, align 16
  %1 = load <16 x i8>, <16 x i8>* @vsc, align 16
  store <16 x i8> %0, <16 x i8>* %__a.addr.i, align 16
  store <16 x i8> %1, <16 x i8>* %__b.addr.i, align 16
  %2 = load <16 x i8>, <16 x i8>* %__a.addr.i, align 16
  %3 = load <16 x i8>, <16 x i8>* %__b.addr.i, align 16
  %4 = call <2 x i64> @llvm.ppc.altivec.vbpermq(<16 x i8> %2, <16 x i8> %3)
  store <2 x i64> %4, <2 x i64>* @res_vll, align 16
  ret void
; CHECK-LABEL: @test1
; CHECK: lvx [[REG1:[0-9]+]],
; CHECK: lvx [[REG2:[0-9]+]],
; CHECK: vbpermq {{[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-VSX: vbpermq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test2() {
entry:
  %__a.addr.i = alloca <16 x i8>, align 16
  %__b.addr.i = alloca <16 x i8>, align 16
  %0 = load <16 x i8>, <16 x i8>* @vuc, align 16
  %1 = load <16 x i8>, <16 x i8>* @vuc, align 16
  store <16 x i8> %0, <16 x i8>* %__a.addr.i, align 16
  store <16 x i8> %1, <16 x i8>* %__b.addr.i, align 16
  %2 = load <16 x i8>, <16 x i8>* %__a.addr.i, align 16
  %3 = load <16 x i8>, <16 x i8>* %__b.addr.i, align 16
  %4 = call <2 x i64> @llvm.ppc.altivec.vbpermq(<16 x i8> %2, <16 x i8> %3)
  store <2 x i64> %4, <2 x i64>* @res_vull, align 16
  ret void
; CHECK-LABEL: @test2
; CHECK: lvx [[REG1:[0-9]+]],
; CHECK: lvx [[REG2:[0-9]+]],
; CHECK: vbpermq {{[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-VSX: vbpermq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test3() {
entry:
  %__a.addr.i = alloca <16 x i8>, align 16
  %0 = load <16 x i8>, <16 x i8>* @vsc, align 16
  store <16 x i8> %0, <16 x i8>* %__a.addr.i, align 16
  %1 = load <16 x i8>, <16 x i8>* %__a.addr.i, align 16
  %2 = call <16 x i8> @llvm.ppc.altivec.vgbbd(<16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @res_vsc, align 16
  ret void
; CHECK-LABEL: @test3
; CHECK: lvx [[REG1:[0-9]+]],
; CHECK: vgbbd {{[0-9]+}}, [[REG1]]
; CHECK-VSX: vgbbd {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test4() {
entry:
  %__a.addr.i = alloca <16 x i8>, align 16
  %0 = load <16 x i8>, <16 x i8>* @vuc, align 16
  store <16 x i8> %0, <16 x i8>* %__a.addr.i, align 16
  %1 = load <16 x i8>, <16 x i8>* %__a.addr.i, align 16
  %2 = call <16 x i8> @llvm.ppc.altivec.vgbbd(<16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @res_vuc, align 16
  ret void
; CHECK-LABEL: @test4
; CHECK: lvx [[REG1:[0-9]+]],
; CHECK: vgbbd {{[0-9]+}}, [[REG1]]
; CHECK-VSX: vgbbd {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vbpermq(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vgbbd(<16 x i8>)
