; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=core-avx2 -mattr=avx2 | FileCheck %s

define <16 x i16> @test_x86_avx2_packssdw(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vpackssdw
  %res = call <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32> %a0, <8 x i32> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32>, <8 x i32>) nounwind readnone


define <32 x i8> @test_x86_avx2_packsswb(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpacksswb
  %res = call <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16> %a0, <16 x i16> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_packuswb(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpackuswb
  %res = call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %a0, <16 x i16> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_padds_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpaddsb
  %res = call <32 x i8> @llvm.x86.avx2.padds.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.padds.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_padds_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpaddsw
  %res = call <16 x i16> @llvm.x86.avx2.padds.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.padds.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_paddus_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpaddusb
  %res = call <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_paddus_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpaddusw
  %res = call <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_pavg_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpavgb
  %res = call <32 x i8> @llvm.x86.avx2.pavg.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pavg.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_pavg_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpavgw
  %res = call <16 x i16> @llvm.x86.avx2.pavg.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pavg.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_pcmpeq_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpcmpeqb
  %res = call <32 x i8> @llvm.x86.avx2.pcmpeq.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pcmpeq.b(<32 x i8>, <32 x i8>) nounwind readnone


define <8 x i32> @test_x86_avx2_pcmpeq_d(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vpcmpeqd
  %res = call <8 x i32> @llvm.x86.avx2.pcmpeq.d(<8 x i32> %a0, <8 x i32> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pcmpeq.d(<8 x i32>, <8 x i32>) nounwind readnone


define <16 x i16> @test_x86_avx2_pcmpeq_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpcmpeqw
  %res = call <16 x i16> @llvm.x86.avx2.pcmpeq.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pcmpeq.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_pcmpgt_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpcmpgtb
  %res = call <32 x i8> @llvm.x86.avx2.pcmpgt.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pcmpgt.b(<32 x i8>, <32 x i8>) nounwind readnone


define <8 x i32> @test_x86_avx2_pcmpgt_d(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vpcmpgtd
  %res = call <8 x i32> @llvm.x86.avx2.pcmpgt.d(<8 x i32> %a0, <8 x i32> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pcmpgt.d(<8 x i32>, <8 x i32>) nounwind readnone


define <16 x i16> @test_x86_avx2_pcmpgt_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpcmpgtw
  %res = call <16 x i16> @llvm.x86.avx2.pcmpgt.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pcmpgt.w(<16 x i16>, <16 x i16>) nounwind readnone


define <8 x i32> @test_x86_avx2_pmadd_wd(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpmaddwd
  %res = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %a0, <16 x i16> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmaxs_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpmaxsw
  %res = call <16 x i16> @llvm.x86.avx2.pmaxs.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmaxs.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_pmaxu_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpmaxub
  %res = call <32 x i8> @llvm.x86.avx2.pmaxu.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pmaxu.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmins_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpminsw
  %res = call <16 x i16> @llvm.x86.avx2.pmins.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmins.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_pminu_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpminub
  %res = call <32 x i8> @llvm.x86.avx2.pminu.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pminu.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmulh_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpmulhw
  %res = call <16 x i16> @llvm.x86.avx2.pmulh.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmulh.w(<16 x i16>, <16 x i16>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmulhu_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpmulhuw
  %res = call <16 x i16> @llvm.x86.avx2.pmulhu.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmulhu.w(<16 x i16>, <16 x i16>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmulu_dq(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vpmuludq
  %res = call <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32> %a0, <8 x i32> %a1) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32>, <8 x i32>) nounwind readnone


define <4 x i64> @test_x86_avx2_psad_bw(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpsadbw
  %res = call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %a0, <32 x i8> %a1) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>) nounwind readnone


define <8 x i32> @test_x86_avx2_psll_d(<8 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpslld
  %res = call <8 x i32> @llvm.x86.avx2.psll.d(<8 x i32> %a0, <4 x i32> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.psll.d(<8 x i32>, <4 x i32>) nounwind readnone


define <4 x i64> @test_x86_avx2_psll_dq(<4 x i64> %a0) {
  ; CHECK: vpslldq
  %res = call <4 x i64> @llvm.x86.avx2.psll.dq(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psll.dq(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psll_dq_bs(<4 x i64> %a0) {
  ; CHECK: vpslldq
  %res = call <4 x i64> @llvm.x86.avx2.psll.dq.bs(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psll.dq.bs(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psll_q(<4 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpsllq
  %res = call <4 x i64> @llvm.x86.avx2.psll.q(<4 x i64> %a0, <2 x i64> %a1) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psll.q(<4 x i64>, <2 x i64>) nounwind readnone


define <16 x i16> @test_x86_avx2_psll_w(<16 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsllw
  %res = call <16 x i16> @llvm.x86.avx2.psll.w(<16 x i16> %a0, <8 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psll.w(<16 x i16>, <8 x i16>) nounwind readnone


define <8 x i32> @test_x86_avx2_pslli_d(<8 x i32> %a0) {
  ; CHECK: vpslld
  %res = call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %a0, i32 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_pslli_q(<4 x i64> %a0) {
  ; CHECK: vpsllq
  %res = call <4 x i64> @llvm.x86.avx2.pslli.q(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pslli.q(<4 x i64>, i32) nounwind readnone


define <16 x i16> @test_x86_avx2_pslli_w(<16 x i16> %a0) {
  ; CHECK: vpsllw
  %res = call <16 x i16> @llvm.x86.avx2.pslli.w(<16 x i16> %a0, i32 7) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pslli.w(<16 x i16>, i32) nounwind readnone


define <8 x i32> @test_x86_avx2_psra_d(<8 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpsrad
  %res = call <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32> %a0, <4 x i32> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32>, <4 x i32>) nounwind readnone


define <16 x i16> @test_x86_avx2_psra_w(<16 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsraw
  %res = call <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16> %a0, <8 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16>, <8 x i16>) nounwind readnone


define <8 x i32> @test_x86_avx2_psrai_d(<8 x i32> %a0) {
  ; CHECK: vpsrad
  %res = call <8 x i32> @llvm.x86.avx2.psrai.d(<8 x i32> %a0, i32 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.psrai.d(<8 x i32>, i32) nounwind readnone


define <16 x i16> @test_x86_avx2_psrai_w(<16 x i16> %a0) {
  ; CHECK: vpsraw
  %res = call <16 x i16> @llvm.x86.avx2.psrai.w(<16 x i16> %a0, i32 7) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psrai.w(<16 x i16>, i32) nounwind readnone


define <8 x i32> @test_x86_avx2_psrl_d(<8 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpsrld
  %res = call <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32> %a0, <4 x i32> %a1) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32>, <4 x i32>) nounwind readnone


define <4 x i64> @test_x86_avx2_psrl_dq(<4 x i64> %a0) {
  ; CHECK: vpsrldq
  %res = call <4 x i64> @llvm.x86.avx2.psrl.dq(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrl.dq(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psrl_dq_bs(<4 x i64> %a0) {
  ; CHECK: vpsrldq
  %res = call <4 x i64> @llvm.x86.avx2.psrl.dq.bs(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrl.dq.bs(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psrl_q(<4 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpsrlq
  %res = call <4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64> %a0, <2 x i64> %a1) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64>, <2 x i64>) nounwind readnone


define <16 x i16> @test_x86_avx2_psrl_w(<16 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsrlw
  %res = call <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16> %a0, <8 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16>, <8 x i16>) nounwind readnone


define <8 x i32> @test_x86_avx2_psrli_d(<8 x i32> %a0) {
  ; CHECK: vpsrld
  %res = call <8 x i32> @llvm.x86.avx2.psrli.d(<8 x i32> %a0, i32 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.psrli.d(<8 x i32>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psrli_q(<4 x i64> %a0) {
  ; CHECK: vpsrlq
  %res = call <4 x i64> @llvm.x86.avx2.psrli.q(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrli.q(<4 x i64>, i32) nounwind readnone


define <16 x i16> @test_x86_avx2_psrli_w(<16 x i16> %a0) {
  ; CHECK: vpsrlw
  %res = call <16 x i16> @llvm.x86.avx2.psrli.w(<16 x i16> %a0, i32 7) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psrli.w(<16 x i16>, i32) nounwind readnone


define <32 x i8> @test_x86_avx2_psubs_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpsubsb
  %res = call <32 x i8> @llvm.x86.avx2.psubs.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.psubs.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_psubs_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpsubsw
  %res = call <16 x i16> @llvm.x86.avx2.psubs.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psubs.w(<16 x i16>, <16 x i16>) nounwind readnone


define <32 x i8> @test_x86_avx2_psubus_b(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vpsubusb
  %res = call <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8> %a0, <32 x i8> %a1) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8>, <32 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_psubus_w(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpsubusw
  %res = call <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16> %a0, <16 x i16> %a1) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16>, <16 x i16>) nounwind readnone
