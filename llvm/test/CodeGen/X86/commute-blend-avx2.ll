; RUN: llc -O3 -mtriple=x86_64-unknown -mcpu=core-avx2 -mattr=avx2 < %s | FileCheck %s

define <8 x i16> @commute_fold_vpblendw_128(<8 x i16> %a, <8 x i16>* %b) #0 {
  %1 = load <8 x i16>, <8 x i16>* %b
  %2 = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %1, <8 x i16> %a, i8 17)
  ret <8 x i16> %2

  ;LABEL:      commute_fold_vpblendw_128
  ;CHECK:      vpblendw {{.*#+}} xmm0 = xmm0[0],mem[1,2,3],xmm0[4],mem[5,6,7]
  ;CHECK-NEXT: retq
}
declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i8) nounwind readnone

define <16 x i16> @commute_fold_vpblendw_256(<16 x i16> %a, <16 x i16>* %b) #0 {
  %1 = load <16 x i16>, <16 x i16>* %b
  %2 = call <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16> %1, <16 x i16> %a, i8 17)
  ret <16 x i16> %2

  ;LABEL:      commute_fold_vpblendw_256
  ;CHECK:      vpblendw {{.*#+}} ymm0 = ymm0[0],mem[1,2,3],ymm0[4],mem[5,6,7],ymm0[8],mem[9,10,11],ymm0[12],mem[13,14,15]
  ;CHECK-NEXT: retq
}
declare <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16>, <16 x i16>, i8) nounwind readnone

define <4 x i32> @commute_fold_vpblendd_128(<4 x i32> %a, <4 x i32>* %b) #0 {
  %1 = load <4 x i32>, <4 x i32>* %b
  %2 = call <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32> %1, <4 x i32> %a, i8 1)
  ret <4 x i32> %2

  ;LABEL:      commute_fold_vpblendd_128
  ;CHECK:      vpblendd {{.*#+}} xmm0 = xmm0[0],mem[1,2,3]
  ;CHECK-NEXT: retq
}
declare <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <8 x i32> @commute_fold_vpblendd_256(<8 x i32> %a, <8 x i32>* %b) #0 {
  %1 = load <8 x i32>, <8 x i32>* %b
  %2 = call <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32> %1, <8 x i32> %a, i8 129)
  ret <8 x i32> %2

  ;LABEL:      commute_fold_vpblendd_256
  ;CHECK:      vpblendd {{.*#+}} ymm0 = ymm0[0],mem[1,2,3,4,5,6],ymm0[7]
  ;CHECK-NEXT: retq
}
declare <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32>, <8 x i32>, i8) nounwind readnone

define <4 x float> @commute_fold_vblendps_128(<4 x float> %a, <4 x float>* %b) #0 {
  %1 = load <4 x float>, <4 x float>* %b
  %2 = call <4 x float> @llvm.x86.sse41.blendps(<4 x float> %1, <4 x float> %a, i8 5)
  ret <4 x float> %2

  ;LABEL:      commute_fold_vblendps_128
  ;CHECK:      vblendps {{.*#+}} xmm0 = xmm0[0],mem[1],xmm0[2],mem[3]
  ;CHECK-NEXT: retq
}
declare <4 x float> @llvm.x86.sse41.blendps(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @commute_fold_vblendps_256(<8 x float> %a, <8 x float>* %b) #0 {
  %1 = load <8 x float>, <8 x float>* %b
  %2 = call <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float> %1, <8 x float> %a, i8 7)
  ret <8 x float> %2

  ;LABEL:      commute_fold_vblendps_256
  ;CHECK:      vblendps {{.*#+}} ymm0 = ymm0[0,1,2],mem[3,4,5,6,7]
  ;CHECK-NEXT: retq
}
declare <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

define <2 x double> @commute_fold_vblendpd_128(<2 x double> %a, <2 x double>* %b) #0 {
  %1 = load <2 x double>, <2 x double>* %b
  %2 = call <2 x double> @llvm.x86.sse41.blendpd(<2 x double> %1, <2 x double> %a, i8 1)
  ret <2 x double> %2

  ;LABEL:      commute_fold_vblendpd_128
  ;CHECK:      vblendpd {{.*#+}} xmm0 = xmm0[0],mem[1]
  ;CHECK-NEXT: retq
}
declare <2 x double> @llvm.x86.sse41.blendpd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @commute_fold_vblendpd_256(<4 x double> %a, <4 x double>* %b) #0 {
  %1 = load <4 x double>, <4 x double>* %b
  %2 = call <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double> %1, <4 x double> %a, i8 7)
  ret <4 x double> %2

  ;LABEL:      commute_fold_vblendpd_256
  ;CHECK:      vblendpd {{.*#+}} ymm0 = ymm0[0,1,2],mem[3]
  ;CHECK-NEXT: retq
}
declare <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone
